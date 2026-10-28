"""Claim extraction and NLI-based claim verification with provenance.

Each answer sentence (and each ';'-separated part) is a claim. Sentences that
only report missing evidence ("The documents do not state ...") are recorded
as gap statements, not factual claims. Each claim is checked against evidence
spans (single sentences, adjacent sentence pairs, whole chunks and the joint
top two spans) with an NLI cross-encoder:

  supported            entailment >= threshold on some span, and every number in
                       the claim appears in that span
  partially_supported  not supported as a whole, but at least one clause is
  contradicted         contradiction >= threshold on some span
  unsupported          otherwise

A supporting span shows the passage entails the claim; it does not prove the
generator causally relied on it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import models
from .config import Settings
from .textutil import numbers, split_sentences
from .trust import Candidate

CITE = re.compile(r"\[\s*((?:E\d+)(?:\s*[,;]\s*E\d+)*)\s*\]")
GAP = re.compile(
    r"\b(do(?:es)? not (?:state|mention|specify|say|provide|include|contain|address)|not (?:stated|mentioned|specified|"
    r"provided|available|found) in|no information|cannot (?:be )?(?:determined|answer|find)|unable to|"
    r"insufficient evidence|not enough evidence)\b",
    re.I,
)
_CLAUSE = re.compile(r",\s*(?:and|but|while|whereas)\s+|\s+(?:but|whereas)\s+|;\s*", re.I)


@dataclass
class Claim:
    kid: str
    text: str
    cited: list[str]
    kind: str = "factual"          # factual | gap
    relation: str | None = None
    best_chunk: str | None = None
    span: str | None = None
    probs: dict = field(default_factory=dict)
    contradicting: dict | None = None
    citations: list[dict] = field(default_factory=list)
    clauses: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def extract_claims(answer: str) -> list[Claim]:
    claims: list[Claim] = []
    # "... delay. [E1]" -> "... delay [E1]." so the citation stays with its sentence.
    answer = re.sub(r"([.!?])\s*((?:\[\s*E\d+(?:\s*[,;]\s*E\d+)*\s*\]\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", answer)
    for sent in split_sentences(answer):
        for part in [p for p in re.split(r";\s+", sent) if p.strip()]:
            cited = []
            for m in CITE.finditer(part):
                cited += [c.strip() for c in re.split(r"[,;]", m.group(1))]
            # Citations at the end of a sentence belong to every part of it.
            if not cited:
                for m in CITE.finditer(sent):
                    cited += [c.strip() for c in re.split(r"[,;]", m.group(1))]
            text = CITE.sub("", part).strip()
            text = re.sub(r"\s+([.,;:])", r"\1", text).strip()
            if len(re.findall(r"[A-Za-z]{2,}", text)) < 3:
                continue
            kind = "gap" if GAP.search(text) else "factual"
            claims.append(Claim(f"K{len(claims) + 1}", text, list(dict.fromkeys(cited)), kind))
    return claims


def _spans(c: Candidate) -> list[str]:
    sents = split_sentences(c.chunk.text)
    spans = list(sents)
    spans += [f"{a} {b}" for a, b in zip(sents, sents[1:])]
    return spans


def _numbers_ok(claim: str, premise: str) -> bool:
    return numbers(claim) <= numbers(premise)


def verify(claims: list[Claim], evidence: list[Candidate], labels: dict[str, str], s: Settings) -> dict:
    """Label every factual claim in place. Returns call statistics."""
    by_id = {c.cid: c for c in evidence}
    factual = [k for k in claims if k.kind == "factual"]
    if not factual:
        return {"nli_pairs": 0}
    span_texts, span_owner = [], []
    for c in evidence:
        for sp in _spans(c):
            span_texts.append(sp)
            span_owner.append(c.cid)
    span_vecs = models.embed(span_texts) if span_texts else None

    pairs, meta = [], []   # meta: (claim index, chunk id or "joint", premise, part)
    for ki, k in enumerate(factual):
        kv = models.embed([k.text])[0]
        chosen: list[tuple[str, str]] = []
        if span_vecs is not None:
            sims = span_vecs @ kv
            order = sims.argsort()[::-1][:8]
            chosen = [(span_owner[i], span_texts[i]) for i in order]
            top_two = []
            for i in order:
                if all(span_owner[i] != o for o, _ in top_two):
                    top_two.append((span_owner[i], span_texts[i]))
                if len(top_two) == 2:
                    break
            if len(top_two) == 2:
                chosen.append(("joint:" + "+".join(o for o, _ in top_two), " ".join(t for _, t in top_two)))
        chosen += [(c.cid, c.chunk.text) for c in evidence]
        for owner, prem in chosen:
            pairs.append((prem, k.text))
            meta.append((ki, owner, prem, None))
        parts = [p.strip(" .") for p in _CLAUSE.split(k.text) if len(p.split()) >= 4]
        if len(parts) >= 2:
            for p in parts:
                for owner, prem in chosen:
                    pairs.append((prem, p))
                    meta.append((ki, owner, prem, p))
    probs = models.nli_probs(pairs)

    for ki, k in enumerate(factual):
        best = (-1.0, None, None, None)      # entail, owner, premise, probs
        worst = (-1.0, None, None)           # contradiction, owner, premise
        per_chunk: dict[str, float] = {}
        clause_best: dict[str, float] = {}
        num_blocked = False
        for (i, owner, prem, part), p in zip(meta, probs):
            if i != ki:
                continue
            if part is not None:
                if _numbers_ok(part, prem):
                    clause_best[part] = max(clause_best.get(part, 0.0), p["entailment"])
                continue
            ent = p["entailment"] if _numbers_ok(k.text, prem) else 0.0
            if ent == 0.0 and p["entailment"] >= s.entail_threshold:
                num_blocked = True
            if ent > best[0]:
                best = (ent, owner, prem, p)
            if not owner.startswith("joint:"):
                per_chunk[owner] = max(per_chunk.get(owner, 0.0), ent)
            if p["contradiction"] > worst[0]:
                worst = (p["contradiction"], owner, prem)
        ent, owner, prem, p = best
        k.best_chunk, k.span = owner, prem
        k.probs = {key: round(v, 3) for key, v in (p or {}).items()}
        if num_blocked and ent < s.entail_threshold:
            k.notes.append("a number in the claim does not appear in the evidence")
        k.clauses = [{"text": t, "entailment": round(v, 3), "supported": v >= s.entail_threshold}
                     for t, v in clause_best.items()]
        if ent >= s.entail_threshold:
            k.relation = "supported"
        elif k.clauses and any(c["supported"] for c in k.clauses):
            k.relation = "partially_supported"
        elif worst[0] >= s.contradict_threshold:
            k.relation = "contradicted"
        else:
            k.relation = "unsupported"
        if worst[0] >= s.contradict_threshold:
            k.contradicting = {"chunk_id": worst[1], "span": worst[2], "contradiction": round(worst[0], 3)}
        if owner and owner.startswith("joint:") and k.relation == "supported":
            k.notes.append("supported only jointly by two passages")
        k.citations = []
        for lab in k.cited:
            cid = labels.get(lab)
            if cid is None or cid not in by_id:
                k.citations.append({"label": lab, "chunk_id": None, "valid": False, "supports": False})
            else:
                sup = per_chunk.get(cid, 0.0) >= s.entail_threshold
                k.citations.append({"label": lab, "chunk_id": cid, "valid": True, "supports": sup})
    return {"nli_pairs": len(pairs)}


def provenance(claims: list[Claim], evidence: list[Candidate]) -> list[dict]:
    by_id = {c.cid: c for c in evidence}
    out = []
    for k in claims:
        if k.kind != "factual":
            continue
        ids = k.best_chunk.split(":", 1)[1].split("+") if k.best_chunk and k.best_chunk.startswith("joint:") else [k.best_chunk]
        for cid in ids:
            c = by_id.get(cid)
            if c is None:
                continue
            out.append({
                "claim_id": k.kid, "claim": k.text, "chunk_id": cid, "doc_id": c.chunk.doc_id,
                "doc_version": c.chunk.version, "locator": c.chunk.locator(), "relation": k.relation,
                "evidence_span": k.span, "entailment": k.probs.get("entailment"),
            })
        if k.contradicting:
            c = by_id.get(k.contradicting["chunk_id"].split(":", 1)[-1].split("+")[0]) if k.contradicting["chunk_id"] else None
            if c is not None:
                out.append({
                    "claim_id": k.kid, "claim": k.text, "chunk_id": c.cid, "doc_id": c.chunk.doc_id,
                    "doc_version": c.chunk.version, "locator": c.chunk.locator(), "relation": "contradicts",
                    "evidence_span": k.contradicting["span"], "contradiction": k.contradicting["contradiction"],
                })
    return out
