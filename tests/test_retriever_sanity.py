import json
from pathlib import Path

import pytest

from core.expl.config import config
from core.expl.utils import _ensure_chain, get_retrieval
from tests.utils import prepare_test

PROJECT_ROOT = Path(__file__).resolve().parents[0]
TEST_DATA_PATH = PROJECT_ROOT / Path("data/text/test_retrieval_augmented_newses.jsonl")
TEST_POOL_PATH = PROJECT_ROOT / Path("data/text/test_retrieval_pool.jsonl")
TEST_VECTOR_PATH = PROJECT_ROOT / Path("data/chroma_db")




@pytest.mark.parametrize("doc", prepare_test(config, TEST_DATA_PATH, TEST_POOL_PATH, TEST_VECTOR_PATH))
def test_at_least_one_retrieval(doc):
    _ensure_chain(config)

    question = doc.page_content
    retrievals = get_retrieval(question)

    assert retrievals is not None
    assert len(retrievals) > 0, "검색 결과가 하나도 존재하지 않음"

    first_retrieval = retrievals[0]
    content = getattr(first_retrieval, "page_content", None) or str(first_retrieval)

    assert isinstance(content, str)
    assert len(content.strip()) > 10, "첫 문서 내용이 너무 짧거나 비어 있음"
