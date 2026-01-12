from datetime import datetime
from typing import Any, List

from eval.interfaces import Step
from eval.prompts.builder import get_prompt_builder
from eval.types import RunContext


class PrepareMetaStep(Step):
    name: str = "prepare_meta"

    def run(self, context: RunContext) -> RunContext:
        context.meta["sample_id"] = context.sample.sample_id
        context.meta["created_at"] = datetime.now().strftime("%Y%m%d-%H%M%S")

        return context
    
    def batch_run(self, contexts): 
        batch_contexts = list()
        for context in contexts: 
            context = self.run(context)
            batch_contexts.append(context)
        
        return batch_contexts


class RetrieveStep(Step):
    name: str = "retrieve"

    retriever: Any   # task에서 주입
    top_k: int
    score_threshold: float
    search_type: str

    def __init__(self, retriever, top_k, score_threshold, search_type):
        self.retriever = retriever

        self.top_k = top_k
        self.score_threshold = score_threshold 
        self.search_type = search_type  

    def run(self, context: RunContext) -> RunContext:
        docs = self.retriever.invoke(
            context.sample.input,
        )

        context.artifacts["retrieved_docs"] = docs
        context.artifacts["retrieval_meta"] = {
            "top_k": self.top_k, 
            "score_threshold": self.score_threshold,
            "search_type": self.search_type
        }

        return context
    
    def batch_run(self, contexts):
        batch_contexts = list()
        for context in contexts:
            context = self.run(context)
            batch_contexts.append(context)
        
        return batch_contexts


class GenerateStep(Step):
    generator: Any
    prompt_builder: Any
    name: str = "generate"
    def __init__(
        self, generator, temperature, 
        top_p, max_tokens, stop, 
        reasoning_effort, prompt_path
    ):
        self.generator = generator 
        self.temperature = temperature 
        self.top_p = top_p 
        self.max_tokens = max_tokens 
        self.stop = stop 
        self.reasoning_effort = reasoning_effort 
        self.prompt_path = prompt_path 

        self.prompt_builder = get_prompt_builder(prompt_path)

    def run(self, context: RunContext) -> RunContext:
        prompt_template = self.prompt_builder.get_template() 
        gen_chain = prompt_template | self.generator

        variables = self.prompt_builder.get_variables(context)
        resp = gen_chain.invoke(variables)

        context.artifacts["prompt"] = prompt_template.format(**variables)
        context.output = resp.text 
        context.meta.update({
            "temperature": self.temperature, 
            "top_p": self.top_p,
            "max_tokens": self.max_tokens, 
            "stop": self.stop, 
            "reasoning_effort": self.reasoning_effort, 
            "prompt_path": self.prompt_path
        })

        return context
    
    def batch_run(self, contexts):
        batch_contexts = list() 
        for context in contexts:
            context = self.run(context)
            batch_contexts.append(context)
        
        return batch_contexts


class Pipeline: 
    steps: List[Step]
    context: RunContext

    def __init__(self, steps):
        self.steps = steps 

    def run(self, context):
        for step in self.steps: 
            context = step.run(context)
        
        return context

    def batch_run(self, batch_contexts):
        for step in self.steps: 
            batch_contexts = self.step.batch_run(batch_contexts)
        
        return batch_contexts

    def add_step(self, step, step_id=None):
        if step_id is None: 
            self.steps.append(step)
        else: 
            self.steps.insert(step, step_id)
    
    def delete_step(self, step_id=None):
        if step_id is None: 
            self.steps.pop()
        else:
            self.steps.pop(step_id)
        