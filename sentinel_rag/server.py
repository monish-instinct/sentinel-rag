"""FastAPI service and static web UI."""
from __future__ import annotations

import json
import threading
import traceback
from collections import deque
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from . import evaluate, generate, models
from .config import DATA_DIR, RELIABILITY_POLICY, Settings
from .demo import load_demo
from .ingest import IngestError
from .pipeline import CONDITIONS, Run
from .store import CorpusStore

WEB = Path(__file__).resolve().parent / "web"
app = FastAPI(title="Sentinel-RAG")
store = CorpusStore()
recent: deque = deque(maxlen=50)
AUDIT = DATA_DIR / "audit.jsonl"
state = {"models_ready": False, "model_error": None,
         "eval": {"running": False, "done": 0, "total": 0, "current": None, "error": None}}


class NoCache(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp


app.add_middleware(NoCache)


def _warm():
    try:
        models.warm_up()
        state["models_ready"] = True
    except Exception as e:  # noqa: BLE001
        state["model_error"] = str(e)


@app.on_event("startup")
def startup():
    if not store.docs:
        try:
            load_demo(store)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    threading.Thread(target=_warm, daemon=True).start()


@app.get("/api/status")
def status():
    return {"models_ready": state["models_ready"], "model_error": state["model_error"],
            "corpus": store.stats(), "generators": generate.describe(), "default_generator": generate.default_name(),
            "conditions": {k: v.label for k, v in CONDITIONS.items()},
            "settings": Settings().to_dict(), "config_hash": Settings().config_hash(),
            "reliability_categories": list(RELIABILITY_POLICY) + ["unknown"]}


@app.get("/api/documents")
def documents():
    docs = sorted(store.docs.values(), key=lambda d: d["doc_id"])
    return {"documents": docs, "chunks": [c.to_dict() for c in store.chunks]}


@app.post("/api/documents")
async def upload(file: UploadFile = File(...), reliability: str = Form("unknown"), published: str = Form(""),
                 source_group: str = Form(""), title: str = Form("")):
    data = await file.read()
    try:
        rec = store.add_document(file.filename or "upload.txt", data,
                                 {"reliability": reliability, "published": published,
                                  "source_group": source_group, "title": title})
    except IngestError as e:
        raise HTTPException(400, str(e)) from e
    return rec


@app.delete("/api/documents/{doc_id}")
def delete(doc_id: str):
    if not store.remove(doc_id):
        raise HTTPException(404, "no such document")
    return {"removed": doc_id}


@app.post("/api/corpus/demo")
def reload_demo():
    return {"loaded": [d["doc_id"] for d in load_demo(store)]}


@app.post("/api/corpus/clear")
def clear():
    store.reset()
    return {"cleared": True}


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    condition: str = "sentinel"
    generator: str | None = None
    model: str | None = None
    top_k: int | None = None
    candidate_pool: int | None = None
    max_rounds: int | None = None


@app.post("/api/ask")
def api_ask(body: AskBody):
    try:
        s = Settings().with_overrides(top_k=body.top_k, candidate_pool=body.candidate_pool, max_rounds=body.max_rounds)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    name = body.generator or generate.default_name()
    try:
        run = Run(store, body.question, body.condition, name, s)
        run.gen = generate.get(name, body.model)
        rec = run.execute()
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    recent.appendleft({"id": rec["id"], "question": rec["question"], "condition": rec["condition"],
                       "decision": rec["decision"]})
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


@app.get("/api/recent")
def api_recent():
    return list(recent)


class EvalBody(BaseModel):
    generator: str = "extractive"
    conditions: list[str] | None = None
    split: str = "all"


@app.post("/api/eval")
def api_eval(body: EvalBody):
    ev = state["eval"]
    if ev["running"]:
        raise HTTPException(409, "an evaluation is already running")
    bad = [c for c in (body.conditions or []) if c not in CONDITIONS]
    if bad:
        raise HTTPException(400, f"unknown conditions {bad}")

    def work():
        ev.update(running=True, done=0, total=0, current=None, error=None)
        try:
            evaluate.run(store, body.generator, body.conditions, body.split,
                         progress=lambda i, n, c, q: ev.update(done=i, total=n, current=f"{c} {q}"))
        except Exception as e:  # noqa: BLE001
            ev["error"] = str(e)
            traceback.print_exc()
        finally:
            ev["running"] = False

    threading.Thread(target=work, daemon=True).start()
    return {"started": True}


@app.get("/api/eval/status")
def eval_status():
    return state["eval"]


@app.get("/api/eval/latest")
def eval_latest():
    res = evaluate.latest()
    return res or JSONResponse({"detail": "no evaluation has been run yet"}, status_code=404)


@app.get("/api/eval/questions")
def eval_questions():
    return evaluate.load_questions("all")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
