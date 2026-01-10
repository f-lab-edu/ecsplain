from pathlib import Path

import pytest

from core.expl.config import config
from core.expl.utils import _ensure_chain, get_answer
from tests.utils import prepare_test

PROJECT_ROOT = Path(__file__).resolve().parents[0]
TEST_DATA_PATH = PROJECT_ROOT / Path(
    "data/retrieval_augmented_newses/20251209-043118/36f808d5eb334206b61ac60e3bbc7e68.jsonl"
)
TEST_POOL_PATH = PROJECT_ROOT / Path(
    "data/retrieval_pool/20251209-043118/36f808d5eb334206b61ac60e3bbc7e68/retrieval_pool.jsonl"
)
TEST_VECTOR_PATH = PROJECT_ROOT / Path("data/chroma_db")


@pytest.mark.parametrize(
    "doc", prepare_test(config, TEST_DATA_PATH, TEST_POOL_PATH, TEST_VECTOR_PATH)
)
def test_rag_e2e_basic(doc):
    _ensure_chain(config)

    question = doc.page_content
    answer, source = get_answer(question)

    assert answer is not None, "답변이 None"

    if isinstance(answer, str):
        answer = answer.strip()  
    else:
        ValueError(f'answer의 타입 {type(answer)}가 지원되지 않습니다.')      

    assert len(answer) > 0, "답변이 비어 있음"
