"""All tunable parameters in one place.

The values follow Table II of the Sentinel-RAG article. Anything the article
left open (chunk units, sufficiency coefficients, verifier thresholds) is
resolved here and documented so every run records exactly what it used.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SENTINEL_DATA_DIR", ROOT / "data"))

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


def _load_dotenv(path: Path = ROOT / ".env"):
    """Read KEY=VALUE lines from the git-ignored .env file (real environment wins)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# Reviewed source categories -> reliability S in [0, 1].
# "unknown" is deliberately not in this table: it is recorded as missing and
# handled by Settings.missing_reliability instead of being trusted by default.
RELIABILITY_POLICY = {
    "official": 1.0,
    "peer_reviewed": 0.85,
    "institutional": 0.75,
    "news": 0.55,
    "secondary": 0.5,
    "blog": 0.3,
    "unverified": 0.2,
}


@dataclass(frozen=True)
class Settings:
    # Chunking. The article's provisional "500 / 50" has unresolved units and
    # all-MiniLM-L6-v2 truncates at 256 word pieces, so chunks are measured in
    # that model's word pieces and capped below the limit.
    chunk_tokens: int = 200
    overlap_tokens: int = 40

    # Retrieval. Candidate pool and final context are separate parameters.
    candidate_pool: int = 15
    top_k: int = 5
    max_rounds: int = 3          # shared retrieval budget M (pre- and post-generation)
    duplicate_cosine: float = 0.95

    # Trust score T = wR*R + wS*S + wC*C + wF*F  (Eq. 1)
    w_relevance: float = 0.40
    w_reliability: float = 0.30
    w_consistency: float = 0.20
    w_freshness: float = 0.10
    missing_reliability: float = 0.4   # unknown source: below neutral, never rewarded
    missing_freshness: float = 0.25    # undated source when the question needs recency
    consistency_topic_cosine: float = 0.5  # passages this similar are compared for agreement
    conflict_threshold: float = 0.7        # NLI contradiction prob. that marks a conflict

    # Sufficiency E = a*T + b*COV + c*AGR + d*IND with bands from the article.
    s_trust: float = 0.30
    s_coverage: float = 0.40
    s_agreement: float = 0.15
    s_independence: float = 0.15
    band_high: float = 0.80
    band_medium: float = 0.60
    coverage_overlap: float = 0.6     # IDF-weighted share of facet terms a passage must contain

    # Claim verification (NLI cross-encoder).
    entail_threshold: float = 0.5
    contradict_threshold: float = 0.6

    # Generation record.
    temperature: float = 0.0
    seed: int | None = 7

    def weights(self) -> dict:
        return {
            "relevance": self.w_relevance,
            "reliability": self.w_reliability,
            "consistency": self.w_consistency,
            "freshness": self.w_freshness,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["embed_model"] = EMBED_MODEL
        d["nli_model"] = NLI_MODEL
        d["reliability_policy"] = RELIABILITY_POLICY
        return d

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def with_overrides(self, **kw) -> "Settings":
        known = {k: v for k, v in kw.items() if k in self.__dataclass_fields__ and v is not None}
        s = replace(self, **known)
        total = s.w_relevance + s.w_reliability + s.w_consistency + s.w_freshness
        if abs(total - 1.0) > 1e-6 or min(s.w_relevance, s.w_reliability, s.w_consistency, s.w_freshness) < 0:
            raise ValueError("trust weights must be non-negative and sum to 1")
        if s.top_k < 1 or s.candidate_pool < s.top_k or s.max_rounds < 1:
            raise ValueError("need 1 <= top_k <= candidate_pool and max_rounds >= 1")
        return s


DEFAULT = Settings()
