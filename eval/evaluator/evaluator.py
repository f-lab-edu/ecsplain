from typing import Any, List, Literal

from core.expl.build import build_generator
from eval.config.eval import EvaluatorConfigUnion
from eval.interfaces import PromptBuilder
from eval.prompts.builder import get_prompt_builder
from eval.types import EvalResult, MetricResult


class LLMEvaluator:
    config: EvaluatorConfigUnion
    name: str
    type: Literal['llm_eval']
    prompt_builder: PromptBuilder
    max_retries: int = 2

    llm: Any = None  # ChatOpenAI 등 (invoke 가능한 객체)

    def __init__(self, config, llm):
        self.config = config 
        self.name = config.name
        self.max_retries = config.max_retries

        self.llm = llm
       
        self.prompt_builder = get_prompt_builder(
            config.gen_component_config.runtime.prompt_path
        )     

    def unit_evaluate(self, context) -> EvalResult:
        if self.llm is None: 
            raise ValueError("llm이 초기화되지 않았습니다.")

        prompt_template = self.prompt_builder.get_template()
        variables = self.prompt_builder.get_variables(context)
        eval_chain = prompt_template | self.llm

        last_err = None 
        for attemt in range(self.max_retries + 1):
            try: 
                resp = eval_chain.invoke(variables).text
                last_err = None 
                break 
            except Exception as e: 
                last_err = repr(e)
        
        if last_err is not None: 
            metric = MetricResult(
                name = self.name, 
                score = 0.0, 
            )

            return EvalResult(
                name = self.name, 
                metric = metric,
                details = {
                    'error': 'judge_generate_failed',
                    'last_err': last_err, 
                }
            )

        try:
            score = resp.split(':')[-1].strip()
        except Exception as e: 
            last_err = repr(e)
            metric = MetricResult(
                name = self.name, 
                score = 0.0
            )

            return EvalResult(
                name = self.name,
                metric = metric,
                details = {
                    'error': 'judge_parse_failed',
                    'last_err': last_err, 
                }
            )
        
        metric = MetricResult(
            name = self.name, 
            score = score, 
        )
        return EvalResult(
            name = self.name,
            metric = metric,
            details = {
                'variables': variables ,
                'judge': resp 
            }
        )
    
    def batch_evaluate(self, contexts) -> List[EvalResult]:
        evaluations = list() 
        for context in contexts:
            evaluations.append(self.unit_evaluate(context))

        return evaluations

def build_evaluator(config):
    if config.type == 'llm_eval':
        llm = build_generator(config.gen_component_config)
        return LLMEvaluator(config, llm)

    raise ValueError(f'{config.type}은 현재 지원되지 않습니다.')

