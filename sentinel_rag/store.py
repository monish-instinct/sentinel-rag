"""Versioned corpus store: document manifest, chunks and a persisted FAISS index.

Similarity convention: embeddings are L2-normalised and search is an exact
inner product over all vectors, so scores are cosine similarities. This is the
same computation as faiss.IndexFlatIP; FAISS itself is not used because its
OpenMP runtime deadlocks alongside PyTorch's on macOS arm64, and exact search is
instant at course-corpus scale. The vectors are persisted in vectors.npy.
"""
from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import models
from .config import DATA_DIR, RELIABILITY_POLICY, Settings
from .ingest import Chunk, IngestError, chunk_blocks, content_hash, parse, slugify
from .textutil import idf_table

SIMILARITY = "cosine (L2-normalised vectors, exact inner product)"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CorpusStore:
    def __init__(self, root: Path | None = None, settings: Settings | None = None):
        self.root = Path(root or DATA_DIR / "corpus")
        self.settings = settings or Settings()
        self.lock = threading.RLock()
        self.docs: dict[str, dict] = {}
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 384), dtype="float32")
        self._by_id: dict[str, int] = {}
        self._idf: dict[str, float] | None = None
        self._load()

    # ------------------------------------------------------------ persistence
    @property
    def _manifest(self) -> Path:
        return self.root / "manifest.json"

    def _load(self):
        if not self._manifest.exists():
            return
        m = json.loads(self._manifest.read_text())
        self.docs = m["documents"]
        self.chunks = [Chunk(**{k: v for k, v in c.items() if k != "locator"}) for c in m["chunks"]]
        vec_path = self.root / "vectors.npy"
        if vec_path.exists() and self.chunks:
            self.vectors = np.load(vec_path)
        self._rebuild()

    def _save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        m = {
            "similarity": SIMILARITY,
            "embed_model": self.settings.to_dict()["embed_model"],
            "chunk_tokens": self.settings.chunk_tokens,
            "overlap_tokens": self.settings.overlap_tokens,
            "updated_at": _now(),
            "documents": self.docs,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        tmp = self._manifest.with_suffix(".tmp")
        tmp.write_text(json.dumps(m, indent=1))
        tmp.replace(self._manifest)
        np.save(self.root / "vectors.npy", self.vectors)

    def _rebuild(self):
        self._by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self._idf = None

    # ------------------------------------------------------------ documents
    def add_document(self, filename: str, data: bytes, meta: dict | None = None) -> dict:
        meta = dict(meta or {})
        reliability = (meta.get("reliability") or "unknown").strip().lower()
        if reliability != "unknown" and reliability not in RELIABILITY_POLICY:
            raise IngestError(f"unknown reliability category '{reliability}'")
        published = (meta.get("published") or "").strip() or None
        if published:
            try:
                datetime.fromisoformat(published)
            except ValueError as e:
                raise IngestError("published date must be ISO format, e.g. 2025-06-01") from e
        doc_id = meta.get("doc_id") or slugify(filename)
        digest = content_hash(data)
        with self.lock:
            prev = self.docs.get(doc_id)
            if prev and prev["content_hash"] == digest:
                changed = False
                for key, val in (("reliability", reliability), ("published", published),
                                 ("source_group", meta.get("source_group") or prev["source_group"]),
                                 ("title", meta.get("title") or prev["title"])):
                    if prev.get(key) != val:
                        prev[key] = val
                        changed = True
                if changed:
                    self._save()
                return {**prev, "status": "unchanged content (metadata updated)" if changed else "unchanged"}
            parsed = parse(filename, data)
            version = (prev["version"] + 1) if prev else 1
            new_chunks = chunk_blocks(parsed.blocks, doc_id, version, self.settings)
            if not new_chunks:
                raise IngestError("no extractable text; nothing was indexed" +
                                  (f" ({'; '.join(parsed.warnings)})" if parsed.warnings else ""))
            vecs = models.embed([c.embed_text() for c in new_chunks])
            # Source-version change invalidates the old version's evidence links.
            keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
            self.chunks = [self.chunks[i] for i in keep] + new_chunks
            self.vectors = np.vstack([self.vectors[keep], vecs]) if len(keep) else vecs
            ext = Path(filename).suffix.lower()
            files = self.root / "files"
            files.mkdir(parents=True, exist_ok=True)
            (files / f"{doc_id}@v{version}{ext}").write_bytes(data)
            history = (prev or {}).get("history", []) + [
                {"version": version, "content_hash": digest, "ingested_at": _now()}
            ]
            rec = {
                "doc_id": doc_id,
                "title": meta.get("title") or Path(filename).stem.replace("_", " "),
                "filename": filename,
                "version": version,
                "content_hash": digest,
                "reliability": reliability,
                "published": published,
                "source_group": (meta.get("source_group") or doc_id).strip(),
                "synthetic": bool(meta.get("synthetic", False)),
                "extraction": parsed.method,
                "pages": parsed.pages,
                "warnings": parsed.warnings,
                "n_chunks": len(new_chunks),
                "token_lengths": [c.n_tokens for c in new_chunks],
                "ingested_at": _now(),
                "history": history,
            }
            self.docs[doc_id] = rec
            self._rebuild()
            self._save()
            return {**rec, "status": "updated" if prev else "added"}

    def remove(self, doc_id: str) -> bool:
        with self.lock:
            if doc_id not in self.docs:
                return False
            keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep] if keep else np.zeros((0, 384), dtype="float32")
            del self.docs[doc_id]
            self._rebuild()
            self._save()
            return True

    def reset(self):
        with self.lock:
            if self.root.exists():
                shutil.rmtree(self.root)
            self.docs, self.chunks = {}, []
            self.vectors = np.zeros((0, 384), dtype="float32")
            self._rebuild()

    # ------------------------------------------------------------ retrieval
    def search(self, query: str, n: int) -> list[tuple[Chunk, float]]:
        with self.lock:
            if not self.chunks:
                return []
            q = models.embed([query])[0]
            scores = self.vectors @ q
            order = np.argsort(-scores, kind="stable")[: min(n, len(self.chunks))]
            return [(self.chunks[i], float(scores[i])) for i in order]

    def chunk(self, chunk_id: str) -> Chunk | None:
        i = self._by_id.get(chunk_id)
        return self.chunks[i] if i is not None else None

    def vector(self, chunk_id: str) -> np.ndarray:
        return self.vectors[self._by_id[chunk_id]]

    def doc(self, doc_id: str) -> dict:
        return self.docs[doc_id]

    def idf(self) -> dict[str, float]:
        with self.lock:
            if self._idf is None:
                self._idf = idf_table([c.embed_text() for c in self.chunks])
            return self._idf

    def stats(self) -> dict:
        return {
            "documents": len(self.docs),
            "chunks": len(self.chunks),
            "similarity": SIMILARITY,
            "chunk_tokens": self.settings.chunk_tokens,
            "overlap_tokens": self.settings.overlap_tokens,
        }
