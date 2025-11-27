import json
from pathlib import Path

import pytest

from core.expl.config import config
from core.expl.utils import _ensure_chain, get_retrieval
 
GOLDEN_PATH = Path('data/expl/golden.jsonl')

def load_jsonl(golden_path):
    cases = list() 
    with golden_path.open('r', encoding='utf-8') as read_fp: 
        for line in read_fp:
            line = line.strip()
            if not line: 
                continue 
            js = json.loads(line)
            cases.append(js)
    
    return cases

@pytest.mark.parametrize('case', load_jsonl(GOLDEN_PATH))
def test_at_least_one_retrieval(case):
    _ensure_chain(config)

    question = case['source']
    print(question)
    retrievals = get_retrieval(question)

    assert retrievals is not None 
    assert len(retrievals) > 0, '검색 결과가 하나도 존재하지 않음'

    first_retrieval = retrievals[0]
    content = getattr(
        first_retrieval, 'page_content', None
    ) or str(first_retrieval)

    assert isinstance(content, str)
    assert len(content.strip()) > 10, "첫 문서 내용이 너무 짧거나 비어 있음"




