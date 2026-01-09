
from eval.config.task import RAGTaskConfig
from eval.runner.pipeline import Pipeline
from eval.runner.task import RAGTask
from eval.types import RunContext, RunOutput


class TaskRunner:
    task: RAGTask
    pipeline: Pipeline
    strict: bool = False

    def __init__(self, task, strict=False):
        self.task = task 
        self._meta = {
            'task_run_id': self.task.id, 
            'task_name': self.task.name,
            'task_hash': self.task.make_hash()
        }

        effective_config = self.task.effective_config()
        self._meta.update(effective_config)

        self.pipeline = task.build_pipeline()

    def make_context(self, sample):
        context = RunContext(sample=sample)
        context.meta.update(self._meta)        

        return context

    def make_output(self, context):
        output = RunOutput(
            sample = context.sample,
            output = context.output, 
            artifacts = context.artifacts,
            meta = context.meta
        )

        return output 

    def run(self):
        samples = self.task.load_samples() 
        outputs = list()
        for sample in samples:
            context = self.make_context(sample)
            context = self.pipeline.run(context)

            output = self.make_output(context)
            outputs.append(output)
        

        return outputs

    def batch_run(self):
        samples = self.task.load_samples()
        
        batch_contexts = [
          self.make_context(sample)  for sample in samples
        ]
        batch_contexts = self.pipeline.batch_run(batch_contexts)
        
        batch_outputs = [
           self.make_output(context) for context in batch_contexts
        ]

        return batch_outputs

def build_task_runner(config):
    if isinstance(config, RAGTaskConfig):
        rag_task = RAGTask(config)
        return TaskRunner(rag_task)

    raise ValueError(f'{type(config)}는 지원되지 않습니다.')
