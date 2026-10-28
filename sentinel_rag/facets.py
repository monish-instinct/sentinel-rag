"""Question facets and facet coverage.

Facet extraction is rule-based (deterministic and inspectable): a question is
split at coordinated sub-questions, and short follow-ups ("and are there any
exceptions?") inherit the topic of the central facet. Coverage requires more
than keyword matching: every facet-specific cue (e.g. "exception") must be
present via its synonym set, and the IDF-weighted share of the facet's terms
in the passage must reach Settings.coverage_overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .textutil import content_terms, split_sentences, stem, term_in, tokens

QWORDS = r"(?:what|which|who|whom|when|where|why|how|whether|is|are|was|were|does|do|did|can|could|should|will|must|has|have)"
_SPLIT = re.compile(
    rf"\s*(?:;|\?\s+|,?\s+(?:and|as well as|also|plus|but)\s+(?={QWORDS}\b)|,\s+(?={QWORDS}\b))\s*",
    re.I,
)

_PAIR = re.compile(r"^(what|which)\s+(?:are|were)\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+)$", re.I)

# Cue words whose presence in a facet must be matched by a cue in the passage.
CUES = {
    "exception": {"exception", "except", "exempt", "exemption", "waive", "waiver", "unless", "excus", "condon", "condonation", "relax", "relaxation"},
    "penalty": {"penalty", "penalis", "penaliz", "deduct", "deduction", "fine"},
    "deadline": {"deadline", "due", "within", "before", "by"},
    "fee": {"fee", "cost", "charge", "price", "inr", "rs", "rupee"},
    "renew": {"renew", "renewal", "retain", "continu"},
    "eligib": {"eligib", "require", "requirement", "minimum", "qualif"},
}


def _cue_of(term: str) -> str | None:
    for cue, syns in CUES.items():
        if term.startswith(cue) or any(term.startswith(s) for s in syns if len(s) >= 5):
            return cue
    return None


@dataclass
class Facet:
    fid: str
    text: str
    terms: list[str]
    distinctive: list[str]
    central: bool = False

    def to_dict(self) -> dict:
        return {"fid": self.fid, "text": self.text, "terms": self.terms,
                "distinctive": self.distinctive, "central": self.central}


@dataclass
class Coverage:
    fid: str
    covered: bool
    by: list[str] = field(default_factory=list)   # chunk ids that cover it
    best_overlap: float = 0.0
    missing_cues: list[str] = field(default_factory=list)


def extract_facets(question: str) -> list[Facet]:
    q = question.strip().rstrip("?").strip()
    parts = [p.strip(" ,.?") for p in _SPLIT.split(q) if p and p.strip(" ,.?")]
    parts = [p for p in parts if content_terms(p)] or [q]
    # "What are the X and the Y" asks for two things.
    expanded = []
    for p in parts:
        m = _PAIR.match(p)
        if m:
            expanded += [f"{m.group(1)} is the {m.group(2)}", f"{m.group(1)} is the {m.group(3)}"]
        else:
            expanded.append(p)
    parts = expanded
    central_terms = content_terms(parts[0])
    facets = []
    for i, p in enumerate(parts):
        terms = content_terms(p)
        own = [t for t in terms if t not in central_terms]
        cue_terms = {t for t in own if _cue_of(t)}
        if i == 0:
            distinctive = []
        elif len(terms) < 3:
            # Short follow-up ("are there any exceptions?"): inherit the central topic so
            # the facet is searchable; every one of its own words is then mandatory.
            terms = terms + [t for t in central_terms if t not in terms]
            p = f"{p} (re: {parts[0][0].lower() + parts[0][1:]})"
            distinctive = sorted(set(own) | cue_terms)
        else:
            # A self-contained sub-question: only its cue words are mandatory.
            distinctive = sorted(cue_terms)
        facets.append(Facet(f"F{i + 1}", p, terms, distinctive, central=(i == 0)))
    return facets


def _cue_present(term: str, stems: set[str], raw: str) -> bool:
    cue = _cue_of(term)
    if cue is None:
        return term_in(term, stems)
    return any(any(s.startswith(syn) for s in stems) for syn in CUES[cue] | {cue}) or term_in(term, stems)


def facet_overlap(facet: Facet, passage: str, idf: dict[str, float]) -> tuple[float, list[str]]:
    stems = {stem(t) for t in tokens(passage)}
    max_idf = max(idf.values(), default=1.0)
    total = present = 0.0
    for t in facet.terms:
        w = idf.get(t, max_idf)
        total += w
        if _cue_present(t, stems, passage):
            present += w
    missing = [t for t in facet.distinctive if not _cue_present(t, stems, passage)]
    return (present / total if total else 0.0), missing


_VEC_CACHE: dict[str, object] = {}


def _vecs(texts: list[str]):
    from . import models

    todo = [t for t in dict.fromkeys(texts) if t not in _VEC_CACHE]
    if todo:
        for t, v in zip(todo, models.embed(todo)):
            _VEC_CACHE[t] = v
    if len(_VEC_CACHE) > 20000:
        _VEC_CACHE.clear()
    return [_VEC_CACHE[t] for t in texts]


def semantic_match(facet: Facet, passage: str) -> float:
    """Highest cosine between the facet and any sentence of the passage (catches paraphrase)."""
    sents = split_sentences(passage) or [passage]
    fv, *svs = _vecs([facet.text] + sents)
    return max(float(v @ fv) for v in svs)


def coverage(facets: list[Facet], passages: dict[str, str], idf: dict[str, float], threshold: float,
             semantic: float = 0.55) -> list[Coverage]:
    """A facet is covered by a passage when every facet-specific cue is present and either
    the IDF-weighted term overlap reaches the bar or a passage sentence paraphrases it."""
    out = []
    for f in facets:
        cov = Coverage(f.fid, False)
        best_missing: list[str] | None = None
        for cid, text in passages.items():
            ov, missing = facet_overlap(f, text, idf)
            if ov > cov.best_overlap:
                cov.best_overlap = ov
            if best_missing is None or len(missing) < len(best_missing):
                best_missing = missing
            # Follow-up facets inherit the central topic; their own cue is the hard
            # requirement, so the topic-overlap bar is lowered for them.
            bar = threshold if f.central or not f.distinctive else threshold / 3
            if missing:
                continue
            if ov >= bar or semantic_match(f, text) >= semantic:
                cov.by.append(cid)
        cov.covered = bool(cov.by)
        cov.missing_cues = [] if cov.covered else (best_missing or [])
        cov.best_overlap = round(cov.best_overlap, 3)
        out.append(cov)
    return out


def refine_query(facet: Facet) -> str:
    """Describe the missing evidence without asserting an answer."""
    return " ".join(dict.fromkeys(facet.distinctive + facet.terms))
