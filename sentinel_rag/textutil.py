"""Small text helpers shared by ingestion, facets, claims and verification."""
from __future__ import annotations

import math
import re

_ABBREV = r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bvs)(?<!\bNo)(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bSt)"
_SENT_SPLIT = re.compile(_ABBREV + r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

STOPWORDS = set(
    """a an the and or but if then else of to in on at by for with from as is are was were be been being
    do does did doing have has had having it its this that these those there their them they he she his her
    we our you your i me my can could should would will shall may might must not no nor so than too very
    what which who whom whose when where why how whether any all each every some such about into over under
    also just only own same other more most much many per up down out off again further once here both few
    s t don now get got tell please us via within without between during before after above below""".split()
)

NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(%|percent|per cent)?", re.I)


def normalize_whitespace(text: str) -> str:
    """Normalise line breaks and runs of spaces; keep punctuation, numbers and negation."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)          # undo hyphenated line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)      # single newlines inside a paragraph
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def split_sentences(text: str) -> list[str]:
    parts = []
    for block in text.split("\n"):
        block = block.strip()
        if block:
            parts.extend(s.strip() for s in _SENT_SPLIT.split(block) if s.strip())
    return parts


def stem(word: str) -> str:
    w = word.lower()
    for suf, rep in (("ies", "y"), ("ions", "ion"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""), ("ly", "")):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + rep
    return w


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.%][0-9]+)?%?", text.lower())


def content_terms(text: str) -> list[str]:
    seen, out = set(), []
    for t in tokens(text):
        if t in STOPWORDS or len(t) < 2:
            continue
        s = stem(t)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def term_in(term: str, stems: set[str]) -> bool:
    if term in stems:
        return True
    if len(term) >= 6:
        return any(len(s) >= 6 and s[:6] == term[:6] for s in stems)
    return False


def numbers(text: str) -> set[str]:
    out = set()
    for m in NUMBER.finditer(text):
        n = m.group(1).replace(",", "")
        if n.endswith(".0"):
            n = n[:-2]
        out.add(n)
    return out


def idf_table(docs: list[str]) -> dict[str, float]:
    df: dict[str, int] = {}
    for d in docs:
        for t in set(content_terms(d)):
            df[t] = df.get(t, 0) + 1
    n = max(len(docs), 1)
    return {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}
