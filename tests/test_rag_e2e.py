import json
from pathlib import Path

import pytest

from core.expl.config import config
from core.expl.utils import _ensure_chain, get_answer

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

@pytest.mark.parametrize("case", load_jsonl(GOLDEN_PATH))
def test_rag_e2e_basic(case):
    _ensure_chain(config)

    question = case['source']
    answer, source = get_answer(question)
    assert answer is not None, "답변이 None"

    answer = answer.strip()
    assert len(answer) > 0, "답변이 비어 있음"
