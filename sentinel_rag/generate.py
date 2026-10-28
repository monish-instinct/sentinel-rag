"""Provider-independent generation interface.

All generators share one interface, so the comparison conditions can hold
the generator fixed while the evidence policy changes:

* ``extractive``   deterministic, no language model. For each question facet it
                   quotes the best-matching evidence sentence with its citation.
                   Fast and reproducible; it cannot paraphrase or answer LLM-only.
* ``groq``         hosted LLM through Groq's OpenAI-compatible API (default
                   openai/gpt-oss-20b). Needs GROQ_API_KEY in .env.
* ``gemini`` / ``openrouter`` / ``custom``
                   the same adapter pointed at other OpenAI-compatible APIs.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import time
import urllib.request
from dataclasses import dataclass

from . import models
from .config import Settings
from .facets import Coverage, Facet
from .textutil import split_sentences
from .trust import Candidate


class GeneratorUnavailable(Exception):
    pass


class GenerationFailure(Exception):
    """The generation service failed (distinct from an abstention)."""


@dataclass
class Generation:
    text: str
    labels: dict[str, str]   # "E1" -> chunk id
    model: str
    seconds: float
    tokens: int = 0
    prompt: str = ""


SYSTEM_RAG = (
    "You answer questions using ONLY the numbered evidence passages provided. "
    "Write 1-4 short factual sentences. End every sentence with the ID of the passage that supports it, "
    "in square brackets, e.g. [E2]. Cite only IDs that appear in the evidence. "
    "Copy numbers, dates and conditions exactly as written. If passages disagree, prefer the newer official "
    "source and mention the disagreement. If the evidence does not answer part of the question, say "
    "'The documents do not state ...' for that part. Do not use outside knowledge."
)
SYSTEM_LLM_ONLY = "Answer the question in 1-4 short factual sentences."
SYSTEM_REWRITE = (
    "Rewrite the verified statements below into a short answer to the question. Use only these statements, "
    "keep their numbers and qualifiers exactly, and keep each statement's [E#] citation. Do not add anything."
)


def evidence_block(evidence: list[Candidate]) -> tuple[str, dict[str, str]]:
    labels, lines = {}, []
    for i, c in enumerate(evidence, start=1):
        lab = f"E{i}"
        labels[lab] = c.cid
        date = c.doc.get("published") or "undated"
        lines.append(f"[{lab}] ({c.doc['title']}, {c.doc['reliability']}, {date})\n{c.chunk.text}")
    return "\n\n".join(lines), labels


class ExtractiveGenerator:
    name = "extractive"
    model = "extractive (no LLM): best evidence sentence per facet"

    def available(self) -> bool:
        return True

    def generate(self, question: str, evidence: list[Candidate], facets: list[Facet],
                 cov: list[Coverage] | None, settings: Settings) -> Generation:
        t0 = time.time()
        if not evidence:
            raise GeneratorUnavailable("the extractive generator needs evidence; it cannot run LLM-only")
        _, labels = evidence_block(evidence)
        inv = {v: k for k, v in labels.items()}
        covmap = {c.fid: c for c in (cov or [])}
        sents = [(c, s) for c in evidence for s in split_sentences(c.chunk.text)]
        sv = models.embed([s for _, s in sents])
        used: set[str] = set()
        out = []
        for f in facets:
            fv = models.embed([f.text])[0]
            allowed = set(covmap[f.fid].by) if f.fid in covmap and covmap[f.fid].covered else None
            best, best_sc = None, -1.0
            for (c, s), v in zip(sents, sv):
                if allowed is not None and c.cid not in allowed:
                    continue
                if s in used:
                    continue
                sc = 0.7 * float(v @ fv) + 0.3 * c.trust
                if sc > best_sc:
                    best, best_sc = (c, s), sc
            if best is None or (allowed is None and best_sc < 0.45):
                out.append(f"The documents do not state {f.text[0].lower() + f.text[1:]}.")
                continue
            c, s = best
            used.add(s)
            s = s.rstrip()
            if not s.endswith((".", "!", "?")):
                s += "."
            out.append(f"{s} [{inv[c.cid]}]")
        return Generation(" ".join(out), labels, self.model, time.time() - t0)

    def rewrite(self, question: str, statements: list[str], labels: dict[str, str], settings: Settings) -> Generation:
        return Generation(" ".join(statements), labels, self.model, 0.0)


class _ChatGenerator:
    name = "chat"
    model = ""

    def _chat(self, system: str, user: str, settings: Settings) -> tuple[str, int]:
        raise NotImplementedError

    def generate(self, question, evidence, facets, cov, settings) -> Generation:
        t0 = time.time()
        if evidence:
            block, labels = evidence_block(evidence)
            user = f"Evidence:\n{block}\n\nQuestion: {question}\nAnswer with citations:"
            system = SYSTEM_RAG
        else:
            labels, system, user = {}, SYSTEM_LLM_ONLY, f"Question: {question}\nAnswer:"
        text, ntok = self._chat(system, user, settings)
        return Generation(text.strip(), labels, self.model, time.time() - t0, ntok, prompt=user)

    def rewrite(self, question, statements, labels, settings) -> Generation:
        t0 = time.time()
        user = "Question: " + question + "\n\nVerified statements:\n" + "\n".join(f"- {s}" for s in statements) + "\n\nAnswer:"
        text, ntok = self._chat(SYSTEM_REWRITE, user, settings)
        return Generation(text.strip(), labels, self.model, time.time() - t0, ntok, prompt=user)


PROVIDERS = {
    # name: (base URL, API-key env var, default model, other models offered in the UI)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-20b",
             ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash",
               ["gemini-2.5-flash"]),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-oss-20b:free",
                   ["openai/gpt-oss-20b:free"]),
    "custom": (os.environ.get("SENTINEL_LLM_BASE_URL", ""), "SENTINEL_LLM_API_KEY",
               os.environ.get("SENTINEL_LLM_MODEL", ""), []),
}


class ApiGenerator(_ChatGenerator):
    """Integration boundary for a hosted OpenAI-compatible chat-completions API.

    Keys come from environment variables (loaded from the git-ignored .env file).
    Rate-limit responses (HTTP 429) are retried with backoff and are recorded as
    generation failures, never as abstentions, if they persist.
    """

    def __init__(self, provider: str, model: str | None = None):
        if provider not in PROVIDERS:
            raise GeneratorUnavailable(f"unknown provider '{provider}'")
        self.name = provider
        self.base, self.key_env, default, self.models = PROVIDERS[provider]
        self.base = self.base.rstrip("/")
        self.model = model or os.environ.get(f"SENTINEL_{provider.upper()}_MODEL") or default

    @property
    def key(self) -> str:
        return os.environ.get(self.key_env, "")

    def available(self) -> bool:
        return bool(self.base and self.model and self.key)

    def _chat(self, system, user, settings):
        if not self.available():
            raise GeneratorUnavailable(f"{self.name}: set {self.key_env} in .env to use this provider")
        payload = {
            "model": self.model, "temperature": settings.temperature, "max_tokens": 1024,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = "low"   # reasoning tokens count against max_tokens
        if settings.seed is not None and self.name == "groq":
            payload["seed"] = settings.seed
        body = json.dumps(payload).encode()
        last = None
        for attempt in range(5):
            req = urllib.request.Request(f"{self.base}/chat/completions", data=body, method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.key}",
                                                  "User-Agent": "sentinel-rag/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = json.loads(r.read())
                msg = data["choices"][0]["message"].get("content") or ""
                return msg, int(data.get("usage", {}).get("completion_tokens", 0))
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
                if e.code in (429, 500, 502, 503):
                    wait = float(e.headers.get("retry-after") or 2 ** attempt)
                    time.sleep(min(wait, 30))
                    continue
                break
            except Exception as e:  # noqa: BLE001
                last = str(e)
                time.sleep(2 ** attempt)
        raise GenerationFailure(f"{self.name} generation failed: {last}")


_GENERATORS: dict[str, object] = {}


def get(name: str, model: str | None = None):
    key = f"{name}|{model or ''}"
    if key not in _GENERATORS:
        _GENERATORS[key] = ExtractiveGenerator() if name == "extractive" else ApiGenerator(name, model)
    return _GENERATORS[key]


def default_name() -> str:
    pref = os.environ.get("SENTINEL_LLM_PROVIDER", "groq")
    return pref if pref in PROVIDERS and get(pref).available() else "extractive"


def describe() -> list[dict]:
    out = [{"name": "extractive", "model": ExtractiveGenerator.model, "models": [], "available": True,
            "kind": "local, no LLM"}]
    for n in PROVIDERS:
        g = get(n)
        out.append({"name": n, "model": g.model or "(not configured)", "models": g.models,
                    "available": g.available(), "kind": f"hosted API (key: {g.key_env})"})
    return out
