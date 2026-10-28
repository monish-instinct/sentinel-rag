from fastapi.testclient import TestClient

from sentinel_rag import evaluate


def claim(rel, cited=True, supports=True):
    return {"kind": "factual", "relation": rel,
            "citations": [{"label": "E1", "chunk_id": "c", "valid": True, "supports": supports}] if cited else []}


def pred(qid, decision, claims, answer="x", cond="sentinel"):
    return {"qid": qid, "condition": cond, "decision": decision, "answer": answer, "claims": claims, "context": [],
            "coverage": [], "timing": {"total_s": 1.0},
            "counters": {"retrieval_calls": 1, "generator_calls": 1, "generated_tokens": 0, "nli_pairs": 0}}


def test_worked_example_from_the_article():
    # Section IV-C: two supported, one partially supported, one contradicted claim.
    p = [pred("q01", "answer", [claim("supported"), claim("supported"),
                                claim("partially_supported", supports=False), claim("contradicted", supports=False)])]
    c = evaluate.claim_counts(p)
    assert c["S"] / c["C"] == 0.5 and c["P"] / c["C"] == 0.25 and (c["U"] + c["X"]) / c["C"] == 0.25


def test_abstention_is_not_counted_as_perfect_faithfulness():
    p = [pred("q01", "abstain", [], answer=None)]
    assert evaluate.claim_counts(p)["C"] == 0
    assert not evaluate.answered(p[0])


def test_correctness_rubric_uses_required_and_forbidden_values():
    q = {"id": "q12", "required": [["5 percent"]], "forbidden": ["20 percent"]}
    good = pred("q12", "answer", [claim("supported")], answer="The penalty is 5 percent per day.")
    bad = pred("q12", "answer", [claim("supported")], answer="It is 5 percent, or 20 percent per the forum.")
    assert evaluate.is_correct(good, q) and not evaluate.is_correct(bad, q)
    unans = {"id": "q17", "answerable": False}
    assert evaluate.is_correct(pred("q17", "abstain", [], answer=None), unans)
    assert not evaluate.is_correct(pred("q17", "error", [], answer=None), unans)


def test_paired_bootstrap_is_zero_for_identical_systems():
    qs = {"q01": {"id": "q01", "required": [["x"]]}, "q02": {"id": "q02", "required": [["y"]]}}
    a = [pred("q01", "answer", [claim("supported")], "x"), pred("q02", "answer", [claim("supported")], "n")]
    r = evaluate.paired_bootstrap(a, a, qs, "accuracy", n_boot=200)
    assert r["diff"] == 0 and r["ci95"] == [0, 0]


def test_evaluation_run_saves_predictions_before_metrics(demo_store):
    res = evaluate.run(demo_store, "extractive", ["vanilla", "sentinel"], "dev")
    from pathlib import Path

    d = Path(res["run_dir"])
    lines = (d / "predictions.jsonl").read_text().splitlines()
    assert len(lines) == 2 * 8 and (d / "metrics.json").exists() and (d / "config.json").exists()
    m = res["metrics"]["sentinel"]
    assert m["N"] == 8 and 0 <= m["accuracy"] <= 1 and m["precision_at_k"] is not None
    assert res["vs_vanilla"]["sentinel"]["accuracy"]["n_questions"] == 8


def test_api_end_to_end():
    from sentinel_rag import server

    with TestClient(server.app) as client:
        st = client.get("/api/status").json()
        assert st["corpus"]["documents"] >= 9
        r = client.get("/static/app.js")
        assert r.status_code == 200 and "no-cache" in r.headers["cache-control"]
        assert client.get("/").status_code == 200
        rec = client.post("/api/ask", json={"question": "What is the minimum pass grade point?",
                                            "condition": "sentinel", "generator": "extractive"}).json()
        assert rec["decision"] == "answer" and "4" in rec["answer"]
        assert client.post("/api/ask", json={"question": "x", "condition": "nope", "generator": "extractive"}).status_code == 400
        up = client.post("/api/documents", files={"file": ("bus.txt", b"The campus shuttle leaves every 20 minutes.")},
                         data={"reliability": "institutional", "published": "2025-01-01"})
        assert up.status_code == 200 and up.json()["status"] == "added"
        rec = client.post("/api/ask", json={"question": "How often does the campus shuttle leave?",
                                            "condition": "sentinel", "generator": "extractive"}).json()
        assert "20 minutes" in (rec["answer"] or "")
        assert client.post("/api/documents", files={"file": ("x.exe", b"MZ")}).status_code == 400
        assert client.delete("/api/documents/bus").status_code == 200
        assert client.delete("/api/documents/bus").status_code == 404
