import pytest

from sentinel_rag.config import Settings
from sentinel_rag.facets import coverage, extract_facets
from sentinel_rag.trust import Candidate, score, temporal_need


def cands(store, query, n=10):
    out, vecs = [], {}
    from sentinel_rag import models

    qv = models.embed([query])[0]
    for ch, _ in store.search(query, n):
        vecs[ch.chunk_id] = store.vector(ch.chunk_id)
        out.append(Candidate(ch, store.doc(ch.doc_id), float(vecs[ch.chunk_id] @ qv), 1, query))
    return out, vecs


def test_trust_is_the_weighted_sum_and_unknown_is_not_rewarded(demo_store):
    s = Settings()
    cs, vecs = cands(demo_store, "When does orientation week start?")
    score(cs, "When does orientation week start?", s, vecs)
    for c in cs:
        assert c.trust == pytest.approx(0.4 * c.R + 0.3 * c.S + 0.2 * c.C + 0.1 * c.F)
        assert all(0.0 <= v <= 1.0 for v in (c.R, c.S, c.C, c.F))
    unknown = next(c for c in cs if c.doc["reliability"] == "unknown")
    assert unknown.S == s.missing_reliability and "reliability unknown" in unknown.flags
    assert "date missing" in unknown.flags


def test_freshness_prefers_newer_for_current_questions(demo_store):
    q = "What is the late submission penalty per day?"
    assert temporal_need(q)[0] == "current"
    cs, vecs = cands(demo_store, q)
    score(cs, q, Settings(), vecs)
    new = next(c for c in cs if c.chunk.doc_id == "assessment-amendment-2025")
    old = next(c for c in cs if c.chunk.doc_id == "academic-handbook-2023")
    assert new.F > old.F


def test_historical_question_flips_freshness(demo_store):
    q = "What late submission penalty did the 2023 handbook specify?"
    assert temporal_need(q) == ("historical", 2023)
    cs, vecs = cands(demo_store, q)
    score(cs, q, Settings(), vecs)
    new = next(c for c in cs if c.chunk.doc_id == "assessment-amendment-2025")
    old = next(c for c in cs if c.chunk.doc_id == "academic-handbook-2023")
    assert old.F > new.F


def test_copied_source_is_marked_duplicate_not_corroboration(demo_store):
    q = "How much attendance do students need to write exams?"
    cs, vecs = cands(demo_store, q, 15)
    score(cs, q, Settings(), vecs)
    forum = [c for c in cs if c.group == "forum-thread-118"]
    assert len(forum) == 2 and sum(c.duplicate_of is not None for c in forum) == 1


def test_facets_split_requirement_and_exception():
    fs = extract_facets("What is the minimum attendance required, and are there any exceptions?")
    assert len(fs) == 2 and fs[0].central and fs[1].distinctive == ["exception"]
    assert len(extract_facets("What are the hostel fee and the re-examination fee?")) == 2
    assert len(extract_facets("What is the parking fee?")) == 1


def test_exception_facet_needs_exception_evidence(demo_store):
    fs = extract_facets("What is the minimum attendance required to sit the end-semester examination, and are there any exceptions?")
    idf = demo_store.idf()
    rule = demo_store.chunk("academic-handbook-2023@v1#c001").embed_text()
    exc = demo_store.chunk("attendance-condonation-circular-2024@v1#c001").embed_text()
    only_rule = coverage(fs, {"rule": rule}, idf, Settings().coverage_overlap)
    assert only_rule[0].covered and not only_rule[1].covered
    both = coverage(fs, {"rule": rule, "exc": exc}, idf, Settings().coverage_overlap)
    assert both[1].covered and both[1].by == ["exc"]
