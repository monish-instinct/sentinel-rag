import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Never touch the real data/ folder from tests.
os.environ["SENTINEL_DATA_DIR"] = tempfile.mkdtemp(prefix="sentinel-test-")


@pytest.fixture(scope="session")
def demo_store(tmp_path_factory):
    from sentinel_rag.demo import load_demo
    from sentinel_rag.store import CorpusStore

    store = CorpusStore(root=tmp_path_factory.mktemp("corpus"))
    load_demo(store)
    return store


@pytest.fixture()
def empty_store(tmp_path):
    from sentinel_rag.store import CorpusStore

    return CorpusStore(root=tmp_path / "corpus")
