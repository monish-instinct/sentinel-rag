"""Trust-aware ranking: T(d, q) = 0.40R + 0.30S + 0.20C + 0.10F  (Eq. 1).

R  relevance: cosine similarity clipped to [0, 1].
S  source reliability: reviewed category -> RELIABILITY_POLICY; unknown is
   recorded as missing and gets Settings.missing_reliability (never a reward).
C  consistency: NLI agreement with topically matched passages from *other*
   source groups (copies of one source are not corroboration). 0.5 = not compared.
F  freshness relative to the question's temporal need; neutral questions give
   every passage 0.5 so freshness cannot change the ranking.
Contradictions are kept and reported, never silently discarded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from . import models
from .config import RELIABILITY_POLICY, Settings
from .ingest import Chunk
from .facets import CUES
from .textutil import content_terms, split_sentences

_QUALIFIER = re.compile(r"\b(" + "|".join(sorted({w for syns in (CUES["exception"],) for w in syns})) + r")\w*", re.I)


def _same_proposition(a: str, b: str) -> bool:
    """Lexical guard: NLI contradiction only counts between passages about the same thing."""
    ta, tb = set(content_terms(a)), set(content_terms(b))
    if not ta or not tb:
        return False
    shared = sum(1 for t in ta if any(t == u or (len(t) >= 5 and len(u) >= 5 and t[:5] == u[:5]) for u in tb))
    return shared / min(len(ta), len(tb)) >= 0.2


@dataclass
class Candidate:
    chunk: Chunk
    doc: dict
    cosine: float
    round_found: int
    query: str
    R: float = 0.0
    S: float = 0.0
    C: float = 0.5
    F: float = 0.5
    trust: float = 0.0
    flags: list[str] = field(default_factory=list)
    duplicate_of: str | None = None
    key_sentence: str = ""

    @property
    def cid(self) -> str:
        return self.chunk.chunk_id

    @property
    def group(self) -> str:
        return self.doc["source_group"]

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.cid,
            "doc_id": self.chunk.doc_id,
            "title": self.doc["title"],
            "version": self.chunk.version,
            "locator": self.chunk.locator(),
            "source_group": self.group,
            "reliability": self.doc["reliability"],
            "published": self.doc.get("published"),
            "synthetic": self.doc.get("synthetic", False),
            "text": self.chunk.text,
            "cosine": round(self.cosine, 4),
            "R": round(self.R, 4), "S": round(self.S, 4), "C": round(self.C, 4), "F": round(self.F, 4),
            "trust": round(self.trust, 4),
            "flags": self.flags,
            "duplicate_of": self.duplicate_of,
            "round_found": self.round_found,
            "query": self.query,
        }


_CURRENT = re.compile(r"\b(current|currently|latest|now|today|present|presently|still|as of|recent|recently|updated|revised|new|nowadays)\b", re.I)
_HIST = re.compile(r"\b(originally|previously|formerly|used to|in the past|old|earlier|before the)\b", re.I)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_PRESENT = re.compile(r"^\s*(what|which|how|who|when|where|is|are|does|do|can|must)\b.*\b(is|are|does|do|can|must|applies|apply|need|needs|required)\b", re.I)


def temporal_need(question: str) -> tuple[str, int | None]:
    m = _YEAR.search(question)
    if m:
        return "historical", int(m.group(0))
    if _HIST.search(question):
        return "historical", None
    if _CURRENT.search(question):
        return "current", None
    if _PRESENT.search(question):
        return "current", None  # present-tense rule questions ask what applies now
    return "neutral", None


def _year(doc: dict) -> float | None:
    p = doc.get("published")
    if not p:
        return None
    d = date.fromisoformat(p[:10])
    return d.year + (d.timetuple().tm_yday - 1) / 366


def score_freshness(cands: list[Candidate], need: str, year: int | None, s: Settings):
    years = [y for y in (_year(c.doc) for c in cands) if y is not None]
    for c in cands:
        y = _year(c.doc)
        if y is None:
            c.flags.append("date missing")
        if need == "neutral":
            c.F = 0.5
        elif need == "current":
            if y is None:
                c.F = s.missing_freshness
            elif max(years) - min(years) < 1e-9:
                c.F = 1.0
            else:
                c.F = (y - min(years)) / (max(years) - min(years))
        else:  # historical
            if y is None:
                c.F = s.missing_freshness
            elif year is None:
                c.F = 1.0 - (y - min(years)) / (max(years) - min(years) + 1e-9)
            else:
                c.F = max(0.0, 1.0 - abs(y - (year + 0.5)) / 5.0)


def score_reliability(cands: list[Candidate], s: Settings):
    for c in cands:
        cat = c.doc.get("reliability", "unknown")
        if cat in RELIABILITY_POLICY:
            c.S = RELIABILITY_POLICY[cat]
        else:
            c.S = s.missing_reliability
            c.flags.append("reliability unknown")


def mark_duplicates(cands: list[Candidate], vecs: dict[str, np.ndarray], s: Settings):
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda c: -c.cosine):
        for k in kept:
            if float(vecs[c.cid] @ vecs[k.cid]) >= s.duplicate_cosine:
                c.duplicate_of = k.cid
                c.flags.append(f"near-duplicate of {k.cid}")
                break
        else:
            kept.append(c)


def key_sentences(cands: list[Candidate], query: str):
    sents, owners = [], []
    for c in cands:
        for snt in split_sentences(c.chunk.text):
            sents.append(snt)
            owners.append(c)
    if not sents:
        return
    qv = models.embed([query])[0]
    sv = models.embed(sents)
    best: dict[str, tuple[float, str]] = {}
    for snt, owner, v in zip(sents, owners, sv):
        sc = float(v @ qv)
        if owner.cid not in best or sc > best[owner.cid][0]:
            best[owner.cid] = (sc, snt)
    for c in cands:
        c.key_sentence = best.get(c.cid, (0, c.chunk.text[:300]))[1]


def score_consistency(cands: list[Candidate], vecs: dict[str, np.ndarray], s: Settings) -> list[dict]:
    active = [c for c in cands if c.duplicate_of is None]
    pairs, meta = [], []
    for a in active:
        for b in active:
            if a is b or a.group == b.group:
                continue
            if float(vecs[a.cid] @ vecs[b.cid]) < s.consistency_topic_cosine:
                continue
            if not _same_proposition(a.key_sentence, b.key_sentence):
                continue
            pairs.append((a.chunk.text, b.key_sentence))  # does a agree with b's key claim?
            meta.append((a, b))
    probs = models.nli_probs(pairs)
    agg: dict[str, list[float]] = {}
    conflicts: dict[tuple[str, str], dict] = {}
    for (a, b), p in zip(meta, probs):
        agg.setdefault(a.cid, []).append(p["entailment"] - p["contradiction"])
        agg.setdefault(b.cid, []).append(p["entailment"] - p["contradiction"])
        qualifies = bool(_QUALIFIER.search(a.chunk.text) or _QUALIFIER.search(b.key_sentence))
        if qualifies:
            # An exception or condonation qualifies a rule rather than contradicting it.
            agg[a.cid][-1] = agg[b.cid][-1] = 0.0
            continue
        if p["contradiction"] >= s.conflict_threshold:
            key = tuple(sorted((a.cid, b.cid)))
            prev = conflicts.get(key)
            if prev is None or p["contradiction"] > prev["contradiction"]:
                conflicts[key] = {"a": a.cid, "b": b.cid, "claim": b.key_sentence,
                                  "contradiction": round(p["contradiction"], 3)}
    for c in active:
        vals = agg.get(c.cid)
        if vals:
            c.C = float(np.clip(0.5 + 0.5 * np.mean(vals), 0.0, 1.0))
        else:
            c.C = 0.5
            c.flags.append("no independent passage to compare")
    for c in cands:
        if c.duplicate_of:
            c.C = next((d.C for d in cands if d.cid == c.duplicate_of), 0.5)
    for k in conflicts:
        for cid in k:
            cc = next(c for c in cands if c.cid == cid)
            if "in conflict" not in cc.flags:
                cc.flags.append("in conflict")
    return list(conflicts.values())


def score(cands: list[Candidate], question: str, s: Settings, vecs: dict[str, np.ndarray],
          use_trust: bool = True) -> list[dict]:
    """Fill R, S, C, F and trust for every candidate. Returns detected conflicts."""
    for c in cands:
        c.flags = []
        c.duplicate_of = None
        c.R = float(np.clip(c.cosine, 0.0, 1.0))
    mark_duplicates(cands, vecs, s)
    score_reliability(cands, s)
    need, year = temporal_need(question)
    score_freshness(cands, need, year, s)
    key_sentences(cands, question)
    conflicts = score_consistency(cands, vecs, s) if use_trust else []
    for c in cands:
        c.trust = (s.w_relevance * c.R + s.w_reliability * c.S + s.w_consistency * c.C + s.w_freshness * c.F)
    for cf in conflicts:
        a = next(c for c in cands if c.cid == cf["a"])
        b = next(c for c in cands if c.cid == cf["b"])
        hi, lo = (a, b) if a.trust >= b.trust else (b, a)
        cf["preferred"] = hi.cid if hi.trust - lo.trust >= 0.05 else None
        cf["resolved"] = cf["preferred"] is not None
        cf["reason"] = (f"{hi.cid} has higher trust ({hi.trust:.2f} vs {lo.trust:.2f})"
                        if cf["resolved"] else "trust scores too close to prefer either source")
    return conflicts
