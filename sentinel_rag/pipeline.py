"""Comparison conditions and the bounded Sentinel-RAG controller (Algorithm 1).

Every condition shares the corpus, embedder, candidate pool, context size k,
generator and verifier. Conditions differ only in the evidence policy:

  llm_only               question only; claims are checked post hoc against the
                         top-k relevance evidence for assessment, never shown to it
  vanilla                top-k by relevance, one pass
  vanilla_iterative      budget-matched: M untargeted retrieval rounds (one per
                         facet), relevance ranking, one generation
  trust                  composite trust ranking, one pass
  sentinel               trust + sufficiency + adaptive retrieval + verification feedback
  sentinel_no_trust      ablation (a): relevance ranking inside the full controller
  sentinel_no_adaptive   ablation (b): one retrieval round; may qualify or abstain
  sentinel_no_feedback   ablation (c): verification labels never trigger retrieval

Baselines are verified only to measure them; their labels never change their output.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import numpy as np

from . import generate as gen
from . import models
from .config import Settings
from .facets import Facet, coverage, extract_facets, refine_query
from .store import CorpusStore
from .textutil import content_terms
from .trust import Candidate, score, temporal_need
from .verify import Claim, extract_claims, provenance, verify


@dataclass(frozen=True)
class Policy:
    retrieve: bool = True
    trust: bool = False
    controller: bool = False      # sufficiency gating, qualification/abstention
    adaptive: bool = False        # pre-generation refinement rounds
    feedback: bool = False        # verification-triggered re-retrieval
    blind_rounds: bool = False    # budget-matched untargeted rounds
    label: str = ""


CONDITIONS: dict[str, Policy] = {
    "llm_only": Policy(retrieve=False, label="LLM-only"),
    "vanilla": Policy(label="Vanilla RAG"),
    "vanilla_iterative": Policy(blind_rounds=True, label="Vanilla RAG, budget-matched iterative"),
    "trust": Policy(trust=True, label="Trust-aware RAG"),
    "sentinel": Policy(trust=True, controller=True, adaptive=True, feedback=True, label="Sentinel-RAG (full)"),
    "sentinel_no_trust": Policy(controller=True, adaptive=True, feedback=True, label="Ablation: no trust score"),
    "sentinel_no_adaptive": Policy(trust=True, controller=True, label="Ablation: no adaptive retrieval"),
    "sentinel_no_feedback": Policy(trust=True, controller=True, adaptive=True, label="Ablation: no verification feedback"),
}
MAIN_CONDITIONS = ["llm_only", "vanilla", "trust", "sentinel"]


def band(e: float, s: Settings) -> str:
    return "high" if e >= s.band_high else "medium" if e >= s.band_medium else "low"


class Run:
    def __init__(self, store: CorpusStore, question: str, condition: str, generator: str, settings: Settings):
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition '{condition}'")
        if not question or not question.strip():
            raise ValueError("question is empty")
        self.store, self.q0, self.cond, self.s = store, question.strip(), condition, settings
        self.policy = CONDITIONS[condition]
        self.gen = gen.get(generator)
        self.facets: list[Facet] = extract_facets(self.q0)
        self.pool: dict[str, Candidate] = {}
        self.vecs: dict[str, np.ndarray] = {}
        self.conflicts: list[dict] = []
        self.rounds: list[dict] = []
        self.events: list[dict] = []
        self.counters = {"retrieval_calls": 0, "generator_calls": 0, "nli_pairs": 0, "generated_tokens": 0}
        self.timing: dict[str, float] = {}
        self.drafts: list[dict] = []
        self.q0v = models.embed([self.q0])[0]

    # --------------------------------------------------------------- pieces
    def retrieve(self, query: str, trigger: str) -> int:
        self.counters["retrieval_calls"] += 1
        hits = self.store.search(query, self.s.candidate_pool)
        new = 0
        for chunk, cos in hits:
            if chunk.chunk_id in self.pool:
                continue
            # Relevance is always measured against the original question q0,
            # the refined query only widens the search.
            self.vecs[chunk.chunk_id] = self.store.vector(chunk.chunk_id)
            self.pool[chunk.chunk_id] = Candidate(chunk, self.store.doc(chunk.doc_id), 0.0, len(self.rounds) + 1, query)
            new += 1
        if not hits:
            self.events.append({"type": "empty_retrieval", "query": query})
        for c in self.pool.values():
            c.cosine = float(self.vecs[c.cid] @ self.q0v)
        self.rounds.append({"round": len(self.rounds) + 1, "query": query, "trigger": trigger,
                            "hits": len(hits), "new_chunks": new})
        return new

    def rank(self) -> list[Candidate]:
        cands = list(self.pool.values())
        self.conflicts = score(cands, self.q0, self.s, self.vecs, use_trust=self.policy.trust)
        key = (lambda c: -c.trust) if self.policy.trust else (lambda c: -c.cosine)
        return sorted(cands, key=key)

    def select(self, ranked: list[Candidate]) -> tuple[list[Candidate], list]:
        k = self.s.top_k
        dispreferred = {cf["b"] if cf["preferred"] == cf["a"] else cf["a"]
                        for cf in self.conflicts if cf.get("resolved")} if self.policy.trust else set()
        usable = [c for c in ranked if c.duplicate_of is None and c.cid not in dispreferred] if self.policy.controller \
            else ranked
        if not self.policy.controller:
            chosen = usable[:k]
        else:
            idf = self.store.idf()
            pool_cov = coverage(self.facets, {c.cid: c.chunk.embed_text() for c in usable}, idf, self.s.coverage_overlap)
            chosen = []
            for fc in pool_cov:              # facet-aware: best covering passage per facet first
                for c in usable:
                    if c.cid in fc.by and c not in chosen:
                        chosen.append(c)
                        break
            for c in usable:
                if len(chosen) >= k:
                    break
                if c not in chosen:
                    chosen.append(c)
            chosen = sorted(chosen[:k], key=lambda c: ranked.index(c))
        cov = coverage(self.facets, {c.cid: c.chunk.embed_text() for c in chosen}, self.store.idf(), self.s.coverage_overlap)
        return chosen, cov

    def sufficiency(self, chosen: list[Candidate], cov) -> dict:
        s = self.s
        t_bar = float(np.mean([c.trust for c in chosen])) if chosen else 0.0
        cov_frac = sum(c.covered for c in cov) / len(cov) if cov else 0.0
        ids = {c.cid for c in chosen}
        in_ctx = [cf for cf in self.conflicts if cf["a"] in ids and cf["b"] in ids]
        unresolved = [cf for cf in in_ctx if not cf.get("resolved")]
        compared = max(1, sum(1 for a in chosen for b in chosen if a is not b and a.group != b.group) // 2)
        agr = 1.0 - min(1.0, len(in_ctx) / compared) if in_ctx else 1.0
        covering_groups = {self.pool[cid].group for fc in cov for cid in fc.by if cid in self.pool}
        ind = min(1.0, len(covering_groups) / 2.0)
        e = s.s_trust * t_bar + s.s_coverage * cov_frac + s.s_agreement * agr + s.s_independence * ind
        b = band(e, s)
        hard = []
        central = next((c for c in cov if c.fid == "F1"), None)
        if central is not None and not central.covered:
            hard.append("central facet not covered")
            b = "low"
        if unresolved and b == "high":
            hard.append("unresolved contradiction in context")
            b = "medium"
        return {"E": round(e, 4), "band": b, "T_bar": round(t_bar, 4), "COV": round(cov_frac, 4),
                "AGR": round(agr, 4), "IND": round(ind, 4), "hard_checks": hard,
                "unresolved_conflicts": len(unresolved)}

    def draft(self, chosen, cov) -> tuple[gen.Generation, list[Claim]]:
        self.counters["generator_calls"] += 1
        g = self.gen.generate(self.q0, chosen, self.facets, cov, self.s)
        self.counters["generated_tokens"] += g.tokens
        claims = extract_claims(g.text)
        st = verify(claims, chosen, g.labels, self.s)
        self.counters["nli_pairs"] += st["nli_pairs"]
        self.drafts.append({"text": g.text, "model": g.model, "seconds": round(g.seconds, 3),
                            "claims": [k.to_dict() for k in claims]})
        return g, claims

    # --------------------------------------------------------------- main
    def execute(self) -> dict:
        t0 = time.time()
        rec = {"id": uuid.uuid4().hex[:10], "question": self.q0, "condition": self.cond,
               "condition_label": self.policy.label, "generator": self.gen.name, "generator_model": self.gen.model,
               "config_hash": self.s.config_hash(), "facets": [f.to_dict() for f in self.facets],
               "temporal_need": temporal_need(self.q0)[0]}
        try:
            out = self._execute()
        except gen.GeneratorUnavailable as e:
            out = {"decision": "error", "error": {"type": "generator_unavailable", "message": str(e)}}
        except gen.GenerationFailure as e:
            out = {"decision": "error", "error": {"type": "generation_failure", "message": str(e)}}
        self.timing["total_s"] = round(time.time() - t0, 3)
        rec.update(out)
        rec.update({"rounds": self.rounds, "events": self.events, "counters": self.counters,
                    "timing": self.timing, "drafts": self.drafts,
                    "candidates": [c.to_dict() for c in sorted(self.pool.values(), key=lambda c: -c.trust)],
                    "conflicts": self.conflicts})
        return rec

    def _final(self, decision, g, claims, chosen, cov, suff, qualification=None, reason=None) -> dict:
        factual = [k for k in claims if k.kind == "factual"]
        if decision == "answer" and not factual:
            decision, reason = "abstain", reason or "the generator produced no factual claims"
        return {
            "decision": decision,
            "answer": g.text if g else None,
            "qualification": qualification,
            "reason": reason,
            "claims": [k.to_dict() for k in claims],
            "labels": g.labels if g else {},
            "provenance": provenance(claims, chosen),
            "context": [c.cid for c in chosen],
            "coverage": [c.__dict__ for c in cov] if cov else [],
            "sufficiency": suff,
        }

    def _execute(self) -> dict:
        p, s = self.policy, self.s
        if not p.retrieve:
            # LLM-only: generate from the question alone, verify post hoc against relevance top-k.
            t = time.time()
            self.counters["generator_calls"] += 1
            g = self.gen.generate(self.q0, [], self.facets, None, s)
            self.counters["generated_tokens"] += g.tokens
            self.timing["generate_s"] = round(time.time() - t, 3)
            self.retrieve(self.q0, "post-hoc assessment only")
            ranked = self.rank()
            chosen = ranked[: s.top_k]
            claims = extract_claims(g.text)
            self.counters["nli_pairs"] += verify(claims, chosen, {}, s)["nli_pairs"]
            self.drafts.append({"text": g.text, "model": g.model, "claims": [k.to_dict() for k in claims]})
            return self._final("answer", g, claims, chosen, None, None)

        t = time.time()
        if self.retrieve(self.q0, "initial") == 0 and not self.pool:
            return {"decision": "abstain", "reason": "retrieval returned nothing (empty corpus?)", "answer": None,
                    "claims": [], "provenance": [], "context": [], "coverage": [], "sufficiency": None}
        if p.blind_rounds:
            for f in self.facets[: s.max_rounds - 1]:
                self.retrieve(f.text, "budget-matched (untargeted)")
        ranked = self.rank()
        chosen, cov = self.select(ranked)
        suff = self.sufficiency(chosen, cov)
        self.rounds[-1].update({"selected": [c.cid for c in chosen], "sufficiency": suff,
                                "coverage": [c.__dict__ for c in cov]})

        if p.controller:
            tried = {self.q0}
            while (suff["band"] == "low" or not all(c.covered for c in cov)) and p.adaptive \
                    and self.counters["retrieval_calls"] < s.max_rounds:
                gaps = [f for f, c in zip(self.facets, cov) if not c.covered] or self.facets
                query = next((refine_query(f) for f in gaps if refine_query(f) not in tried), None)
                if query is None:
                    self.rounds[-1]["action"] = "stop: every gap query already tried"
                    break
                tried.add(query)
                self.rounds[-1]["action"] = f"refine for uncovered {', '.join(f.fid for f in gaps)}"
                new = self.retrieve(query, "insufficient evidence")
                ranked = self.rank()
                chosen, cov = self.select(ranked)
                suff = self.sufficiency(chosen, cov)
                self.rounds[-1].update({"selected": [c.cid for c in chosen], "sufficiency": suff,
                                        "coverage": [c.__dict__ for c in cov]})
                if new == 0:
                    self.rounds[-1]["action"] = "stop: round added no new evidence"
                    break
            self.timing["retrieve_s"] = round(time.time() - t, 3)
            central = cov[0] if cov else None
            if central is not None and not central.covered:
                self.rounds[-1]["action"] = "abstain before drafting"
                return {"decision": "abstain", "answer": None, "qualification": None,
                        "reason": f"no retained passage covers the central question ({self.facets[0].text}) "
                                  f"after {self.counters['retrieval_calls']} retrieval round(s)",
                        "claims": [], "provenance": [], "context": [c.cid for c in chosen],
                        "coverage": [c.__dict__ for c in cov], "sufficiency": suff}
        else:
            self.timing["retrieve_s"] = round(time.time() - t, 3)

        t = time.time()
        g, claims = self.draft(chosen, cov)
        if not p.controller:
            self.timing["generate_verify_s"] = round(time.time() - t, 3)
            return self._final("answer", g, claims, chosen, cov, suff)

        # Verification feedback: unsupported material claims trigger targeted retrieval.
        while p.feedback and self.counters["retrieval_calls"] < s.max_rounds:
            failing = [k for k in claims if k.kind == "factual" and k.relation != "supported"]
            if not failing:
                break
            terms = [t for t in content_terms(" ".join(k.text for k in failing)) if not any(ch.isdigit() for ch in t)]
            query = " ".join(terms[:16])
            self.rounds[-1]["action"] = f"re-retrieve for unsupported {', '.join(k.kid for k in failing)}"
            new = self.retrieve(query, "verification failed")
            ranked = self.rank()
            chosen, cov = self.select(ranked)
            suff = self.sufficiency(chosen, cov)
            self.rounds[-1].update({"selected": [c.cid for c in chosen], "sufficiency": suff,
                                    "coverage": [c.__dict__ for c in cov]})
            if new == 0:
                self.rounds[-1]["action"] = "stop: round added no new evidence"
                break
            g, claims = self.draft(chosen, cov)

        factual = [k for k in claims if k.kind == "factual"]
        supported = [k for k in factual if k.relation == "supported"]
        missing = [f for f, c in zip(self.facets, cov) if not c.covered]
        unresolved = [cf for cf in self.conflicts
                      if not cf.get("resolved") and cf["a"] in g.labels.values() and cf["b"] in g.labels.values()]
        notes = []
        if missing:
            notes.append("Not established by the indexed documents: " + "; ".join(f.text for f in missing) + ".")
        if unresolved:
            notes.append("The sources disagree and their trust scores are too close to prefer one: "
                         + "; ".join(f"{cf['a']} vs {cf['b']}" for cf in unresolved) + ".")
        resolved = [cf for cf in self.conflicts if cf.get("resolved") and cf["preferred"] in g.labels.values()]
        if resolved:
            notes.append("Conflicting lower-trust evidence was set aside: "
                         + "; ".join(f"{cf['b'] if cf['preferred'] == cf['a'] else cf['a']} ({cf['reason']})" for cf in resolved) + ".")

        if factual and len(supported) == len(factual) and not missing and not unresolved:
            self.timing["generate_verify_s"] = round(time.time() - t, 3)
            return self._final("answer", g, claims, chosen, cov, suff, qualification=" ".join(notes) or None)
        if not supported:
            self.timing["generate_verify_s"] = round(time.time() - t, 3)
            out = self._final("abstain", g, claims, chosen, cov, suff,
                              reason="no generated claim is supported by the retained evidence")
            out["answer"] = None
            out["withheld_draft"] = g.text
            return out

        # Budget reached: regenerate only the supported scope, then verify the delivered text.
        inv = {v: k for k, v in g.labels.items()}
        statements = []
        for k in supported:
            lab = next((c["label"] for c in k.citations if c["supports"]), None) or inv.get(k.best_chunk)
            statements.append(f"{k.text} [{lab}]" if lab else k.text)
        self.counters["generator_calls"] += 1
        rg = self.gen.rewrite(self.q0, statements, g.labels, s)
        self.counters["generated_tokens"] += rg.tokens
        rclaims = extract_claims(rg.text)
        self.counters["nli_pairs"] += verify(rclaims, chosen, rg.labels, s)["nli_pairs"]
        if any(k.kind == "factual" and k.relation != "supported" for k in rclaims):
            rg = gen.Generation(" ".join(statements), g.labels, rg.model + " (fallback: verified statements verbatim)", 0.0)
            rclaims = extract_claims(rg.text)
            self.counters["nli_pairs"] += verify(rclaims, chosen, rg.labels, s)["nli_pairs"]
        dropped = [k.text for k in factual if k.relation != "supported"]
        if dropped:
            notes.insert(0, f"{len(dropped)} draft claim(s) were removed because the evidence did not support them.")
        self.drafts.append({"text": rg.text, "model": rg.model, "stage": "supported-scope rewrite",
                            "claims": [k.to_dict() for k in rclaims]})
        self.timing["generate_verify_s"] = round(time.time() - t, 3)
        out = self._final("qualified", rg, rclaims, chosen, cov, suff, qualification=" ".join(notes) or None)
        out["removed_claims"] = dropped
        return out


def ask(store: CorpusStore, question: str, condition: str = "sentinel", generator: str = "extractive",
        settings: Settings | None = None) -> dict:
    if not question or not question.strip():
        raise ValueError("question is empty")
    return Run(store, question, condition, generator, settings or Settings()).execute()
