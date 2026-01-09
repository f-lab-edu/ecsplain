from typing import List, Protocol

from eval.types import EvalResult, RunContext


class PromptBuilder(Protocol):
    name: str 

    def get_template(self):
        ...

    def get_variables(self, context):
        ...

class Evaluator(Protocol):
    name: str

    def unit_evaluate(self, context) -> EvalResult: 
        ...

    def batch_evaluate(self, contexts) -> List[EvalResult]:
        ...

class Reporter(Protocol):
    name: str 

    def report_one(self, context, report) -> None: 
        ...

class Step(Protocol):
    name: str

    def run(self, ctx: RunContext) -> RunContext: 
        ...

class Task(Protocol):
    name: str

