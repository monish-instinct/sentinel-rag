"""Load the synthetic demo corpus with its declared source metadata."""
from __future__ import annotations

import json

from .config import ROOT
from .store import CorpusStore

DEMO_DIR = ROOT / "demo_corpus"


def load_demo(store: CorpusStore, reset: bool = True) -> list[dict]:
    if reset:
        store.reset()
    manifest = json.loads((DEMO_DIR / "manifest.json").read_text())
    out = []
    for d in manifest["documents"]:
        data = (DEMO_DIR / d["file"]).read_bytes()
        meta = {k: d.get(k) for k in ("title", "reliability", "published", "source_group")}
        meta["synthetic"] = True
        out.append(store.add_document(d["file"], data, meta))
    return out
