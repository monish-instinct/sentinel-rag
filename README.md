# Sentinel-RAG

A working prototype of **Sentinel-RAG: a trust-aware adaptive retrieval framework for provenance-grounded hallucination mitigation** (Sinha and Jagatheshwaran, VIT). It runs locally on a Mac: a Python service plus a browser UI.

It implements the article end to end:

- a **vanilla RAG baseline**: ingestion, chunking, embeddings, vector retrieval, grounded generation and source references.
- **trust-aware ranking**: `T = 0.40R + 0.30S + 0.20C + 0.10F` over relevance, source reliability, cross-source consistency and freshness.
- **evidence sufficiency**: `E = αT̄ + βCOV + γAGR + δIND`, with question-facet coverage, high/medium/low bands and hard checks.
- **claim-level verification** with an NLI model, plus **claim-to-span provenance** (document, version, page/section, evidence span).
- the **bounded feedback loop** (Algorithm 1): refine the query, retrieve again, regenerate only the supported scope, qualify the answer or abstain. Every round shares one retrieval budget.
- the **evaluation protocol**: LLM-only vs vanilla vs trust-aware vs full Sentinel-RAG, a budget-matched iterative baseline, three ablations, all of the article's metrics, and paired bootstrap confidence intervals.

## Run it

```bash
cd ~/Downloads/sentinel-rag
./run.sh
```

The browser opens at <http://127.0.0.1:8000>. The first run creates `.venv`, installs dependencies and downloads two small local models (about 370 MB). Later runs start in a few seconds.

| Command | What it does |
|---|---|
| `./run.sh` | start the app (port 8000; `PORT=8001 ./run.sh` to change) |
| `./run.sh test` | run the automated test-suite (43 tests) |
| `./run.sh eval --generator extractive` | run the full evaluation from the terminal |
| `./run.sh eval --generator groq --conditions llm_only vanilla trust sentinel` | evaluation with the hosted LLM |

### LLM API key

Generation uses a hosted LLM through **Groq's OpenAI-compatible API** (default model `openai/gpt-oss-20b`). Put the key in a `.env` file in the project folder. The file is git-ignored and never committed:

```bash
cp .env.example .env    # then paste your key after GROQ_API_KEY=
```

Gemini, OpenRouter or any other OpenAI-compatible endpoint also work through the same adapter (see `.env.example`). Without a key, the app still runs with the **extractive generator**. That generator uses no LLM: it quotes the best evidence sentence for each question facet, so every part of the pipeline works offline.

The embedder (`all-MiniLM-L6-v2`) and the verifier (`cross-encoder/nli-deberta-v3-xsmall`) always run locally.

## Using the demo

The app starts with a **synthetic demo corpus** in `demo_corpus/`: nine documents about a fictional "Northbridge Institute of Technology". Every file is labelled synthetic. The corpus is built to trigger the failure modes the article describes:

| Case | Documents involved |
|---|---|
| Requirement and its exception under a different heading | Handbook (75 % attendance) + condonation circular ("condone", not "exception") |
| Outdated rule vs amendment | Handbook 2023 (10 %/day) vs Senate amendment 2025 (5 %/day) |
| Low-quality source | Forum post claiming 20 %/day and 60 % attendance |
| Copied source (false corroboration) | Forum post + its mirror, same source group |
| Missing metadata | Orientation notes with unknown reliability and no date |
| PDF page locators / DOCX tables | Lab safety policy (2 pages), scholarship policy (table) |

Things to try:

1. **Ask tab**: click the example chips. Each answer shows every claim coloured by its verification label. Click a claim to see its evidence span, NLI scores, provenance and citation check. Below the answer are the facets and their coverage, the sufficiency score, each retrieval round with the action taken, any detected conflicts, and the candidate pool with R/S/C/F/T scores.
2. **Compare conditions**: the same question through LLM-only, vanilla, trust-aware and Sentinel-RAG side by side. "What is the parking fee for students?" is a good one. Vanilla answers with an off-topic but "supported" scholarship sentence; Sentinel abstains.
3. **Corpus**: upload your own PDF, DOCX, TXT or MD with a declared reliability, date and source group. Uploading a changed file with the same name creates version 2 and invalidates the old evidence links.
4. **Evaluation**: run the 24-question comparison from the UI and read the metric table, bootstrap intervals, per-category and per-question results.
5. **Method**: the formulas and every parameter in use.

## Measured results (synthetic set)

The tables below come from runs on this machine, recorded in `data/runs/` (not committed). The evaluation set is **24 synthetic questions** (`eval/questions.json`). It is small and was written for this prototype. These numbers show the mechanisms behave as designed on controlled cases; they are **not** benchmark evidence of general superiority.

**Run 1: hosted LLM** (Groq `openai/gpt-oss-20b`, temperature 0, main four conditions, 24 questions, 0 service errors)

| condition | acc | ans | sel | abst-U | faith | unsup | cite | P@k | R@k | MRR | lat s | rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 0.25 | 1.00 | 0.25 | 0.00 | 0.15 | 0.80 | n/a | n/a | n/a | n/a | 1.35 | 1 |
| vanilla | 1.00 | 0.79 | 1.00 | 1.00 | 0.88 | 0.08 | 0.96 | 0.24 | 1.00 | 0.92 | 2.44 | 1 |
| trust | 0.96 | 0.75 | 1.00 | 1.00 | 0.92 | 0.00 | 0.91 | 0.23 | 0.95 | 0.86 | 4.25 | 1 |
| sentinel | 0.96 | 0.75 | 1.00 | 1.00 | 1.00 | 0.00 | 0.96 | 0.24 | 1.00 | 0.87 | 3.94 | 1.38 |

| vs vanilla | Δ accuracy [95% CI] | Δ faithfulness | Δ unsupported rate |
|---|---|---|---|
| llm_only | -0.75 [-0.92, -0.58] | -0.73 [-0.88, -0.57] | +0.72 [+0.54, +0.88] |
| trust | -0.04 [-0.12, +0.00] | +0.04 [-0.01, +0.11] | -0.08 [-0.19, +0.00] |
| sentinel | -0.04 [-0.12, +0.00] | +0.12 [+0.00, +0.23] | -0.08 [-0.19, +0.00] |

What this run shows:

- **LLM-only** invents answers about the fictional institute: 80 % of its claims are unsupported by the corpus and it never abstains.
- **Vanilla RAG** already gets every answer right on this small set, because the strong LLM refuses on its own when the context is irrelevant. But 8 % of its claims are unsupported or contradicted by the evidence it was given.
- **Sentinel-RAG** delivers no unsupported claims: faithfulness 1.00 vs 0.88. The 95 % CI of the difference is [0.00, +0.23], so on 24 questions this gain is borderline. It wrongly abstained on one answerable question (q15, "Do students lose 20 percent…?"), where the verifier rejected every drafted claim.
- On accuracy there is **no measurable gain** over vanilla in this run.
- Sentinel costs about 1.6× vanilla's latency and 1.38 retrieval rounds on average.

**Run 2: extractive generator** (no LLM, all conditions including the budget-matched baseline and ablations)

| condition | acc | ans | sel | abst-U | faith | unsup | cite | P@k | R@k | MRR | lat s | rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla | 0.71 | 0.96 | 0.70 | 0.20 | 1.00 | 0.00 | 1.00 | 0.24 | 1.00 | 0.92 | 0.54 | 1 |
| trust | 0.67 | 0.92 | 0.68 | 0.20 | 1.00 | 0.00 | 1.00 | 0.23 | 0.95 | 0.86 | 0.48 | 1 |
| sentinel | 0.88 | 0.79 | 0.84 | 1.00 | 1.00 | 0.00 | 1.00 | 0.24 | 1.00 | 0.87 | 0.45 | 1.25 |
| vanilla_iterative | 0.71 | 0.96 | 0.70 | 0.20 | 1.00 | 0.00 | 1.00 | 0.24 | 1.00 | 0.92 | 0.29 | 2.21 |
| sentinel_no_trust | 0.88 | 0.79 | 0.84 | 1.00 | 1.00 | 0.00 | 1.00 | 0.24 | 1.00 | 0.92 | 0.26 | 1.25 |
| sentinel_no_adaptive | 0.88 | 0.79 | 0.84 | 1.00 | 1.00 | 0.00 | 1.00 | 0.24 | 1.00 | 0.87 | 0.44 | 1 |
| sentinel_no_feedback | 0.88 | 0.79 | 0.84 | 1.00 | 1.00 | 0.00 | 1.00 | 0.24 | 1.00 | 0.87 | 0.53 | 1.25 |

| vs vanilla | Δ accuracy [95% CI] | Δ faithfulness | Δ unsupported rate |
|---|---|---|---|
| trust | -0.04 [-0.12, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| sentinel | +0.17 [+0.04, +0.33] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| vanilla_iterative | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| sentinel_no_trust | +0.17 [+0.04, +0.33] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| sentinel_no_adaptive | +0.17 [+0.04, +0.33] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| sentinel_no_feedback | +0.17 [+0.04, +0.33] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

What this run shows:

- The extractive generator quotes evidence, so faithfulness is 1.00 by construction everywhere. The differences come from **what** is quoted and **when to abstain**.
- Vanilla and trust-aware RAG answer 4 of 5 unanswerable questions with off-topic sentences.
- Sentinel's facet coverage and sufficiency check abstain on all 5 unanswerable questions, which raises accuracy from 0.71 to 0.88 at a lower answer rate (0.79).
- The ablations barely differ from full Sentinel here. With a deterministic generator, the verification loop has almost nothing to correct, and this set has few cases where trust re-ranking changes the selected context. Trust-aware RAG alone loses one answer (q23), because it ranks the undated, unknown-reliability orientation notes below other passages.
- Remaining failures:
  - q13: the outdated "three days" rule is quoted because that conflict is not detected.
  - q16: a historical question is answered from the amendment.
  - q24: the curfew sentence is quoted instead of the visiting hours.

Metric definitions follow Section III-E of the article. Accuracy counts an abstention on an answerable question as wrong and an abstention on an unanswerable question as right. Claim metrics (faithfulness, unsupported rate, citation correctness) count only answered questions; an abstention is never scored as perfectly faithful. P@k is low by construction: most questions have a single relevant chunk and k = 5.

## How it maps to the article

| Article | Code |
|---|---|
| Pre-processing (III-C): PDF/DOCX/TXT parsing, normalisation, chunking, IDs, versions | `sentinel_rag/ingest.py`, `sentinel_rag/store.py` |
| Vanilla RAG baseline (III-A) | `pipeline.py`, condition `vanilla` |
| Trust score, Eq. (1) | `trust.py` |
| Facets, coverage and sufficiency (III-B2) | `facets.py`, `Run.sufficiency` in `pipeline.py` |
| Bounded feedback algorithm (III-B3, Fig. 5/6) | `Run._execute` in `pipeline.py` |
| Claim extraction, verification, provenance (Fig. 7) | `verify.py` |
| Provider-independent generation (III-D) | `generate.py` |
| Evaluation conditions, ablations, metrics (IV) | `pipeline.CONDITIONS`, `evaluate.py` |

Parameters (Table II) live in `sentinel_rag/config.py`, and every run records them with a config hash. Choices the article left open are resolved and documented there:

- **Chunk units.** 200 word pieces of the embedder's own tokenizer with 40 of overlap. The model truncates at 256, so the article's "500" would silently lose text; the code refuses to index any chunk that would be truncated.
- **Sufficiency coefficients.** α = 0.30, β = 0.40, γ = 0.15, δ = 0.15 (provisional).
- **Verifier thresholds.** Entailment ≥ 0.5 means supported, contradiction ≥ 0.6 means contradicted. A claim is only supported if every number in it appears in the evidence span.
- **Missing metadata.** Unknown reliability gets S = 0.4 and is flagged. A missing date gets F = 0.25 when the question needs recency. Neither is rewarded.

## Known limitations (honest list)

- **Small synthetic evaluation.** Weights, thresholds and coefficients are the article's provisional values or engineering choices, not calibrated on a development set. No public benchmark (HotpotQA, ALCE) has been run. There is no human annotation; correctness uses a keyword rubric over required and forbidden facts.
- **Rule-based facets.** Facet extraction is rule-based. It handles coordinated sub-questions ("…, and are there any exceptions?", "the X and the Y") but not implicit constraints. Coverage is lexical (IDF-weighted, with cue-word synonyms) plus a semantic sentence match, so it can miss unusual paraphrases.
- **NLI verifier limits.** The verifier is a small NLI model. It can mislabel claims about numbers, tables, long spans or multi-passage reasoning. Entailment shows a passage *supports* a claim; it does not prove the LLM *used* it (no causal attribution).
- **Conflict detection is heuristic.** NLI contradiction between passages about the same proposition, with a lexical same-topic guard, and with exception or condonation language treated as qualification rather than contradiction. Some real conflicts are missed. For example, the handbook's "three days" vs the amendment's "five days" is not flagged, and the extractive generator then quotes the outdated rule.
- **Source reliability is declared, not inferred.** It is only as good as the metadata entered at upload.
- **Exact search instead of FAISS.** The article suggests FAISS or Chroma. FAISS's OpenMP runtime deadlocks next to PyTorch's on macOS arm64, so search is an exact inner product over normalised vectors in NumPy. That is the same computation as `faiss.IndexFlatIP`, and instant at this corpus size. It would need a real ANN index for large corpora.
- **No OCR.** Image-only PDF pages are reported as warnings and not indexed.
- **Hosted LLM variability.** The hosted LLM can change behind the same model name, and Groq's free tier rate-limits requests (retries with backoff are built in; persistent failures are reported as errors, never as abstentions). Temperature 0 and a fixed seed reduce, but do not remove, run-to-run variation.
- **Local single-user app.** No authentication. It is meant to run on your own machine.

## Project layout

```
run.sh                 one-command start / test / eval
sentinel_rag/          the Python package (config, ingest, store, trust, facets,
                       generate, verify, pipeline, evaluate, server, web/ UI)
demo_corpus/           synthetic documents + manifest with declared metadata
eval/questions.json    24-question synthetic evaluation set (dev/test split)
scripts/make_demo_docs.py   regenerates the demo PDF and DOCX
tests/                 pytest suite
data/                  created at runtime: corpus index, audit log, eval runs (git-ignored)
```
