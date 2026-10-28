import pytest

from sentinel_rag.ingest import IngestError
from sentinel_rag.store import CorpusStore


def test_versioning_invalidates_old_chunks(empty_store):
    s = empty_store
    r1 = s.add_document("policy.txt", b"The fee is INR 100 per semester.", {"reliability": "official"})
    assert r1["status"] == "added" and r1["version"] == 1
    assert s.add_document("policy.txt", b"The fee is INR 100 per semester.", {"reliability": "official"})["status"] == "unchanged"
    r2 = s.add_document("policy.txt", b"The fee is INR 250 per semester.", {"reliability": "official"})
    assert r2["version"] == 2 and len(r2["history"]) == 2
    assert all(c.version == 2 for c in s.chunks)
    assert all("v1" not in c.chunk_id for c in s.chunks)
    assert "250" in s.search("semester fee", 1)[0][0].text


def test_metadata_is_validated(empty_store):
    with pytest.raises(IngestError):
        empty_store.add_document("a.txt", b"Some text here for the test.", {"reliability": "very trustworthy"})
    with pytest.raises(IngestError):
        empty_store.add_document("a.txt", b"Some text here for the test.", {"published": "June 2025"})


def test_unknown_reliability_is_stored_as_unknown(empty_store):
    rec = empty_store.add_document("a.txt", b"Some text here for the test.")
    assert rec["reliability"] == "unknown" and rec["published"] is None


def test_search_is_cosine_sorted_and_store_persists(tmp_path):
    s = CorpusStore(root=tmp_path / "c")
    s.add_document("a.txt", b"Goggles must be worn in the chemistry lab.", {})
    s.add_document("b.txt", b"The library opens at nine on Saturdays.", {})
    hits = s.search("When does the library open?", 2)
    assert hits[0][0].doc_id == "b" and hits[0][1] >= hits[1][1]
    assert -1.0 <= hits[1][1] <= 1.0
    again = CorpusStore(root=tmp_path / "c")
    assert {c.chunk_id for c in again.chunks} == {c.chunk_id for c in s.chunks}
    assert again.search("When does the library open?", 1)[0][0].doc_id == "b"
    assert again.remove("a") and all(c.doc_id != "a" for c in again.chunks)
