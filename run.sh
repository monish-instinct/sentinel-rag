#!/usr/bin/env bash
# One-command local start for Sentinel-RAG:  ./run.sh   (then open http://127.0.0.1:8000)
#   ./run.sh test    run the test-suite
#   ./run.sh eval    run the evaluation from the command line (extractive generator)
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
PY=/opt/homebrew/bin/python3.12
command -v "$PY" >/dev/null 2>&1 || PY=python3.12
command -v "$PY" >/dev/null 2>&1 || PY=python3

if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment with $PY ..."
  "$PY" -m venv .venv
fi
if ! .venv/bin/python -c "import fastapi, sentence_transformers, fitz, docx" >/dev/null 2>&1; then
  echo "Installing dependencies (first run only) ..."
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
# Fetch the two local models once (embedder ~90 MB, NLI verifier ~280 MB).
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for repo, pats in [("sentence-transformers/all-MiniLM-L6-v2", None),
                   ("cross-encoder/nli-deberta-v3-xsmall", ["*.json", "model.safetensors", "spm.model", "*.txt"])]:
    try:
        snapshot_download(repo, local_files_only=True, allow_patterns=pats)
    except Exception:
        print(f"Downloading {repo} (first run only) ...")
        snapshot_download(repo, allow_patterns=pats)
PY
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore

case "${1:-serve}" in
  test) exec .venv/bin/python -m pytest -q tests ;;
  eval) shift; exec .venv/bin/python -m sentinel_rag.evaluate "$@" ;;
  serve)
    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "Port $PORT is already in use. Stop the other server or run: PORT=8001 ./run.sh"
      exit 1
    fi
    [ -f .env ] || echo "Note: no .env file, so only the extractive (no-LLM) generator is available. See README."
    echo "Sentinel-RAG on http://127.0.0.1:$PORT  (Ctrl+C to stop)"
    (sleep 3 && open "http://127.0.0.1:$PORT" >/dev/null 2>&1 || true) &
    exec .venv/bin/python -m uvicorn sentinel_rag.server:app --host 127.0.0.1 --port "$PORT" --log-level warning ;;
  *) echo "usage: ./run.sh [serve|test|eval]"; exit 2 ;;
esac
