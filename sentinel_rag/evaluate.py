"""Evaluation harness (Section IV): run every condition on the same questions,
store predictions before scoring, then compute the article's metrics.

Usage:
    .venv/bin/python -m sentinel_rag.evaluate --generator extractive
    .venv/bin/python -m sentinel_rag.evaluate --generator groq --conditions llm_only vanilla trust sentinel
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from .config import DATA_DIR, ROOT, Settings
from .pipeline import CONDITIONS, MAIN_CONDITIONS, ask
from .store import CorpusStore

QUESTIONS = ROOT / "eval" / "questions.json"
RUNS = DATA_DIR / "runs"


def load_questions(split: str = "all") -> list[dict]:
    qs = json.loads(QUESTIONS.read_text())["questions"]
    return [q for q in qs if split == "all" or q["split"] == split]


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    return t.replace("‑", "-").replace("‐", "-").replace("–", "-").replace(" ", " ")


def answered(pred: dict) -> bool:
    return pred.get("decision") in ("answer", "qualified") and any(
        k["kind"] == "factual" for k in pred.get("claims", []))


def is_correct(pred: dict, q: dict) -> bool:
    if not q.get("answerable", True):
        return not answered(pred) and pred.get("decision") != "error"
    if not answered(pred):
        return False
    text = _norm(pred.get("answer"))
    if any(_norm(f) in text for f in q.get("forbidden", [])):
        return False
    return all(any(_norm(alt) in text for alt in group) for group in q.get("required", []))


def relevant_ids(store: CorpusStore, q: dict) -> set[str]:
    out = set()
    for doc_id, phrase in q.get("relevant", []):
        for c in store.chunks:
            if c.doc_id == doc_id and _norm(phrase) in _norm(c.text):
                out.add(c.chunk_id)
    return out


def compact(rec: dict) -> dict:
    """What is kept per prediction (enough to re-score without regenerating)."""
    keep = ("id", "question", "condition", "generator", "generator_model", "config_hash", "decision", "answer",
            "qualification", "reason", "error", "claims", "context", "coverage", "sufficiency", "counters",
            "timing", "conflicts", "removed_claims", "withheld_draft")
    out = {k: rec.get(k) for k in keep}
    out["rounds"] = [{k: r.get(k) for k in ("round", "trigger", "query", "new_chunks", "action")} for r in rec["rounds"]]
    return out


def _ratio(a, b):
    return round(a / b, 4) if b else None


def _pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return round(xs[i], 3)


def claim_counts(preds: list[dict]) -> dict:
    c = {"C": 0, "S": 0, "P": 0, "U": 0, "X": 0, "links": 0, "good_links": 0, "covered_claims": 0}
    for p in preds:
        if not answered(p):
            continue
        for k in p["claims"]:
            if k["kind"] != "factual":
                continue
            c["C"] += 1
            c[{"supported": "S", "partially_supported": "P", "unsupported": "U", "contradicted": "X"}[k["relation"]]] += 1
            c["links"] += len(k["citations"])
            good = sum(1 for l in k["citations"] if l["valid"] and l["supports"])
            c["good_links"] += good
            c["covered_claims"] += 1 if good else 0
    return c


def score_condition(preds: list[dict], qs: dict[str, dict], store: CorpusStore, s: Settings, condition: str) -> dict:
    n = len(preds)
    ans = [p for p in preds if answered(p)]
    correct = [p for p in preds if is_correct(p, qs[p["qid"]])]
    unans = [p for p in preds if not qs[p["qid"]].get("answerable", True)]
    answerable = [p for p in preds if qs[p["qid"]].get("answerable", True)]
    cc = claim_counts(preds)
    # Retrieval metrics: answerable questions with relevance labels; not applicable to LLM-only.
    prec, rec, rr = [], [], []
    if condition != "llm_only":
        for p in answerable:
            rel = relevant_ids(store, qs[p["qid"]])
            if not rel:
                continue
            ctx = p.get("context") or []
            hit = [cid for cid in ctx[: s.top_k] if cid in rel]
            prec.append(len(hit) / s.top_k)
            rec.append(len(hit) / len(rel))
            rank = next((i + 1 for i, cid in enumerate(ctx) if cid in rel), None)
            rr.append(1 / rank if rank else 0.0)
    facet_cov = [sum(c["covered"] for c in p["coverage"]) / len(p["coverage"]) for p in answerable if p.get("coverage")]
    lat = [p["timing"]["total_s"] for p in preds if p.get("timing")]
    has_citations = condition != "llm_only"
    return {
        "condition": condition,
        "label": CONDITIONS[condition].label,
        "N": n,
        "answered": len(ans),
        "answer_rate": _ratio(len(ans), n),
        "accuracy": _ratio(len(correct), n),
        "selective_accuracy": _ratio(sum(1 for p in ans if is_correct(p, qs[p["qid"]])), len(ans)),
        "answerable_accuracy": _ratio(sum(1 for p in answerable if is_correct(p, qs[p["qid"]])), len(answerable)),
        "abstention_on_unanswerable": _ratio(sum(1 for p in unans if not answered(p)), len(unans)),
        "abstention_on_answerable": _ratio(sum(1 for p in answerable if not answered(p)), len(answerable)),
        "errors": sum(1 for p in preds if p.get("decision") == "error"),
        "claims": cc["C"],
        "faithfulness": _ratio(cc["S"], cc["C"]),
        "partial_rate": _ratio(cc["P"], cc["C"]),
        "unsupported_rate": _ratio(cc["U"] + cc["X"], cc["C"]),
        "contradicted_rate": _ratio(cc["X"], cc["C"]),
        "citation_links": cc["links"] if has_citations else None,
        "citation_correctness": _ratio(cc["good_links"], cc["links"]) if has_citations else None,
        "evidence_coverage": _ratio(cc["covered_claims"], cc["C"]) if has_citations else None,
        "retrieval_eligible": len(prec),
        "precision_at_k": round(statistics.mean(prec), 4) if prec else None,
        "recall_at_k": round(statistics.mean(rec), 4) if rec else None,
        "mrr": round(statistics.mean(rr), 4) if rr else None,
        "facet_coverage": round(statistics.mean(facet_cov), 4) if facet_cov else None,
        "latency_mean_s": round(statistics.mean(lat), 3) if lat else None,
        "latency_median_s": _pct(lat, 0.5),
        "latency_p90_s": _pct(lat, 0.9),
        "retrieval_calls_mean": round(statistics.mean(p["counters"]["retrieval_calls"] for p in preds), 3),
        "generator_calls_mean": round(statistics.mean(p["counters"]["generator_calls"] for p in preds), 3),
        "generated_tokens_total": sum(p["counters"]["generated_tokens"] for p in preds),
        "nli_pairs_mean": round(statistics.mean(p["counters"]["nli_pairs"] for p in preds), 1),
    }


def paired_bootstrap(a: list[dict], b: list[dict], qs: dict, metric: str, n_boot: int = 2000, seed: int = 7) -> dict | None:
    """95% CI for metric(b) - metric(a), resampling question IDs (claims stay with their question)."""
    by_a = {p["qid"]: p for p in a}
    by_b = {p["qid"]: p for p in b}
    ids = sorted(set(by_a) & set(by_b))
    if not ids:
        return None

    def value(preds):
        if metric == "accuracy":
            return sum(is_correct(p, qs[p["qid"]]) for p in preds) / len(preds)
        cc = claim_counts(preds)
        if not cc["C"]:
            return None
        return cc["S"] / cc["C"] if metric == "faithfulness" else (cc["U"] + cc["X"]) / cc["C"]

    base_a, base_b = value([by_a[i] for i in ids]), value([by_b[i] for i in ids])
    if base_a is None or base_b is None:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [rng.choice(ids) for _ in ids]
        va, vb = value([by_a[i] for i in sample]), value([by_b[i] for i in sample])
        if va is not None and vb is not None:
            diffs.append(vb - va)
    diffs.sort()
    return {"diff": round(base_b - base_a, 4), "ci95": [round(diffs[int(0.025 * len(diffs))], 4),
                                                        round(diffs[int(0.975 * len(diffs)) - 1], 4)],
            "n_questions": len(ids)}


def score_run(preds: list[dict], store: CorpusStore, s: Settings) -> dict:
    qs = {q["id"]: q for q in load_questions("all")}
    by_cond: dict[str, list[dict]] = {}
    for p in preds:
        by_cond.setdefault(p["condition"], []).append(p)
    metrics = {c: score_condition(ps, qs, store, s, c) for c, ps in by_cond.items()}
    categories = {}
    for c, ps in by_cond.items():
        cats: dict[str, list] = {}
        for p in ps:
            cats.setdefault(qs[p["qid"]]["category"], []).append(p)
        categories[c] = {cat: {"n": len(v), "correct": sum(is_correct(p, qs[p["qid"]]) for p in v),
                               "answered": sum(answered(p) for p in v)} for cat, v in sorted(cats.items())}
    comparisons = {}
    if "vanilla" in by_cond:
        for c, ps in by_cond.items():
            if c == "vanilla":
                continue
            comparisons[c] = {m: paired_bootstrap(by_cond["vanilla"], ps, qs, m)
                              for m in ("accuracy", "faithfulness", "unsupported_rate")}
    per_question = {}
    for p in preds:
        per_question.setdefault(p["qid"], {})[p["condition"]] = {
            "decision": p["decision"], "correct": is_correct(p, qs[p["qid"]]),
            "answer": p.get("answer"), "reason": p.get("reason") or (p.get("error") or {}).get("message")}
    return {"metrics": metrics, "categories": categories, "vs_vanilla": comparisons, "per_question": per_question}


def run(store: CorpusStore, generator: str = "extractive", conditions: list[str] | None = None,
        split: str = "all", settings: Settings | None = None, progress=None) -> dict:
    s = settings or Settings()
    conditions = conditions or (MAIN_CONDITIONS + ["vanilla_iterative", "sentinel_no_trust",
                                                   "sentinel_no_adaptive", "sentinel_no_feedback"])
    if generator == "extractive":
        conditions = [c for c in conditions if c != "llm_only"]   # LLM-only needs a language model
    qs = load_questions(split)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = RUNS / f"{stamp}_{generator}"
    out_dir.mkdir(parents=True, exist_ok=True)
    from . import generate as gen

    config = {"generator": generator, "generator_model": gen.get(generator).model, "conditions": conditions,
              "split": split, "questions": [q["id"] for q in qs], "settings": s.to_dict(),
              "config_hash": s.config_hash(), "corpus": {d: v["content_hash"][:12] for d, v in store.docs.items()},
              "started": stamp}
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))
    preds = []
    total = len(conditions) * len(qs)
    t0 = time.time()
    with (out_dir / "predictions.jsonl").open("w") as fh:
        for c in conditions:                      # fixed order: condition, then question id
            for q in qs:
                rec = compact(ask(store, q["question"], c, generator, s))
                rec["qid"] = q["id"]
                preds.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if progress:
                    progress(len(preds), total, c, q["id"])
    result = score_run(preds, store, s)
    result.update({"run_dir": str(out_dir), "config": config, "seconds": round(time.time() - t0, 1)})
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1))
    (RUNS / "latest.json").write_text(json.dumps({"run_dir": str(out_dir)}))
    return result


def latest() -> dict | None:
    pointer = RUNS / "latest.json"
    if not pointer.exists():
        return None
    run_dir = Path(json.loads(pointer.read_text())["run_dir"])
    m = run_dir / "metrics.json"
    return json.loads(m.read_text()) if m.exists() else None


def _table(result: dict) -> str:
    cols = [("accuracy", "acc"), ("answer_rate", "ans"), ("selective_accuracy", "sel"),
            ("abstention_on_unanswerable", "abst-U"), ("faithfulness", "faith"), ("unsupported_rate", "unsup"),
            ("citation_correctness", "cite"), ("precision_at_k", "P@k"), ("recall_at_k", "R@k"), ("mrr", "MRR"),
            ("latency_mean_s", "lat s"), ("retrieval_calls_mean", "rounds")]
    lines = ["| condition | " + " | ".join(h for _, h in cols) + " |", "|---|" + "---|" * len(cols)]
    for c, m in result["metrics"].items():
        vals = ["n/a" if m[k] is None else (f"{m[k]:.2f}" if isinstance(m[k], float) else str(m[k])) for k, _ in cols]
        lines.append(f"| {c} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Run the Sentinel-RAG comparison on the synthetic question set")
    ap.add_argument("--generator", default="extractive")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--reload-demo", action="store_true", help="reset the corpus to demo_corpus/ first")
    args = ap.parse_args()
    store = CorpusStore()
    if args.reload_demo or not store.docs:
        from .demo import load_demo

        load_demo(store)
    res = run(store, args.generator, args.conditions, args.split,
              progress=lambda i, n, c, q: print(f"\r{i}/{n} {c} {q}   ", end="", flush=True))
    print()
    print(_table(res))
    print("\nPaired bootstrap vs vanilla (diff, 95% CI):")
    for c, comp in res["vs_vanilla"].items():
        print(" ", c, {m: (v["diff"], v["ci95"]) if v else None for m, v in comp.items()})
    print("\nsaved:", res["run_dir"])


if __name__ == "__main__":
    main()
