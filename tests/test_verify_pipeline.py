import pytest

from sentinel_rag import generate as gen
from sentinel_rag.config import Settings
from sentinel_rag.pipeline import CONDITIONS, Run, ask
from sentinel_rag.verify import extract_claims


def test_claim_extraction_keeps_citations_and_separates_gap_statements():
    claims = extract_claims("The fee is INR 1,500 per course. [E2] The documents do not state whether laundry is included.")
    assert [k.kind for k in claims] == ["factual", "gap"]
    assert claims[0].cited == ["E2"] and "[E2]" not in claims[0].text


class Scripted(gen._ChatGenerator):
    """Stands in for a hosted LLM so the feedback loop can be tested offline."""
    name = "scripted"
    model = "scripted-test-model"

    def __init__(self, drafts):
        self.drafts = list(drafts)
        self.calls = 0

    def available(self):
        return True

    def _chat(self, system, user, settings):
        self.calls += 1
        if "Verified statements" in user:   # supported-scope rewrite: echo the statements
            return " ".join(l[2:] for l in user.splitlines() if l.startswith("- ")), 5
        return (self.drafts.pop(0) if self.drafts else "The documents do not state this."), 10


def run_with(store, q, drafts, condition="sentinel", **kw):
    r = Run(store, q, condition, "extractive", Settings().with_overrides(**kw))
    r.gen = Scripted(drafts)
    return r.execute(), r.gen


def test_verifier_labels_supported_contradicted_and_number_mismatch(demo_store):
    q = "What is the late submission penalty per day?"
    rec, _ = run_with(demo_store, q, ["The late submission penalty is 5 percent of the marks for each day of delay. [E1] "
                                      "The late submission penalty is 50 percent of the marks for each day of delay. [E1]"],
                      condition="vanilla")
    k1, k2 = [k for k in rec["claims"] if k["kind"] == "factual"]
    assert k1["relation"] == "supported" and k1["citations"][0]["supports"]
    assert k2["relation"] in ("contradicted", "unsupported")
    assert not k2["citations"][0]["supports"]


def test_invalid_citation_counts_as_failure(demo_store):
    rec, _ = run_with(demo_store, "What is the late submission penalty per day?",
                      ["The late submission penalty is 5 percent of the marks for each day of delay. [E9]"], condition="vanilla")
    cite = rec["claims"][0]["citations"][0]
    assert cite == {"label": "E9", "chunk_id": None, "valid": False, "supports": False}


def test_unsupported_claim_is_removed_and_answer_qualified(demo_store):
    q = "What is the annual hostel fee?"
    rec, g = run_with(demo_store, q, [
        "The annual hostel fee is INR 85,000, which includes accommodation and meals. [E1] Laundry is free for all residents. [E1]",
        "The annual hostel fee is INR 85,000, which includes accommodation and meals. [E1] Laundry is free for all residents. [E1]",
        "The annual hostel fee is INR 85,000, which includes accommodation and meals. [E1] Laundry is free for all residents. [E1]",
    ])
    assert rec["decision"] == "qualified"
    assert "Laundry" not in rec["answer"] and "85,000" in rec["answer"]
    assert rec["removed_claims"] == ["Laundry is free for all residents."]
    assert all(k["relation"] == "supported" for k in rec["claims"] if k["kind"] == "factual")
    triggers = [r["trigger"] for r in rec["rounds"]]
    assert "verification failed" in triggers          # verification fed back into retrieval
    assert rec["counters"]["retrieval_calls"] <= 3      # shared budget M


def test_feedback_ablation_never_retrieves_after_verification(demo_store):
    rec, _ = run_with(demo_store, "What is the annual hostel fee?",
                      ["The annual hostel fee is INR 85,000, which includes accommodation and meals. [E1] Laundry is free. [E1]"],
                      condition="sentinel_no_feedback")
    assert "verification failed" not in [r["trigger"] for r in rec["rounds"]]


def test_all_unsupported_draft_leads_to_abstention(demo_store):
    rec, _ = run_with(demo_store, "What is the annual hostel fee?", ["Residents get a free bicycle every year. [E1]"] * 3)
    assert rec["decision"] == "abstain" and rec["answer"] is None and rec["withheld_draft"]


def test_sentinel_abstains_on_unanswerable_but_vanilla_does_not(demo_store):
    q = "What is the parking fee for students?"
    s = ask(demo_store, q, "sentinel", "extractive")
    v = ask(demo_store, q, "vanilla", "extractive")
    assert s["decision"] == "abstain" and "central question" in s["reason"]
    assert v["decision"] == "answer"
    assert s["counters"]["generator_calls"] == 0     # abstained before drafting


def test_requirement_and_exception_both_answered(demo_store):
    rec = ask(demo_store, "What is the minimum attendance required to sit the end-semester examination, "
                          "and are there any exceptions?", "sentinel", "extractive")
    assert rec["decision"] == "answer"
    assert "75 percent" in rec["answer"] and "condone" in rec["answer"]
    assert all(c["covered"] for c in rec["coverage"])
    assert {p["doc_id"] for p in rec["provenance"]} >= {"academic-handbook-2023", "attendance-condonation-circular-2024"}


def test_partially_answerable_question_is_qualified(demo_store):
    rec = ask(demo_store, "What is the annual hostel fee and does it include laundry?", "sentinel", "extractive")
    assert rec["decision"] == "qualified" and "85,000" in rec["answer"] and "laundry" in rec["qualification"]


def test_trust_condition_prefers_the_amendment_over_the_forum(demo_store):
    rec = ask(demo_store, "What is the late submission penalty per day?", "trust", "extractive")
    assert "5 percent" in rec["answer"] and "20 percent" not in rec["answer"]


@pytest.mark.parametrize("condition", list(CONDITIONS))
def test_every_condition_respects_the_round_budget(demo_store, condition):
    if condition == "llm_only":
        rec = ask(demo_store, "What is the minimum pass grade point?", condition, "extractive")
        assert rec["decision"] == "error" and rec["error"]["type"] == "generator_unavailable"
        return
    rec = ask(demo_store, "What is the annual hostel fee and does it include laundry?", condition, "extractive",
              Settings().with_overrides(max_rounds=2))
    assert rec["counters"]["retrieval_calls"] <= 2
    assert len(rec["context"]) <= Settings().top_k


def test_settings_reject_invalid_weights():
    with pytest.raises(ValueError):
        Settings().with_overrides(w_relevance=0.9)
    with pytest.raises(ValueError):
        Settings().with_overrides(top_k=20)


def test_generation_failure_is_reported_as_error_not_abstention(demo_store):
    class Broken(Scripted):
        def _chat(self, system, user, settings):
            raise gen.GenerationFailure("service down")

    r = Run(demo_store, "What is the minimum pass grade point?", "sentinel", "extractive", Settings())
    r.gen = Broken([])
    rec = r.execute()
    assert rec["decision"] == "error" and rec["error"]["type"] == "generation_failure"
