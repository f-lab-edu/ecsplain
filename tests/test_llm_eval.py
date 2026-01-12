import pytest

from eval.config.eval import get_eval_config
from eval.evaluator.engine import build_eval_engine
from eval.runner.runner import build_task_runner


def rag_generate():
    rag_config = get_eval_config('rag')
    rag_runner = build_task_runner(rag_config)
    run_outputs = rag_runner.run()

    return run_outputs

@pytest.mark.parametrize("run_output", rag_generate())
def test_llm_eval(run_output):
    eval_config = get_eval_config('eval_engine')
    eval_engine = build_eval_engine(eval_config)

    eval_input = run_output.to_eval() 
    report = eval_engine.run(eval_input)

    assert all([
        isinstance(score, int) or isinstance(score, float)  for score in report.scores.values()
        if score is not None
    ])
