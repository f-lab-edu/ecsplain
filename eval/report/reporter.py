from uuid import uuid4

from langsmith import Client


class LangSmithReporter:
    def __init__(self, client: Client, logger=None, strict=False, project_name=''):
        self.client = client
        self.logger = logger
        self.strict = strict

        self.project_name = project_name

    def report_one(self, context, report):
        try:
            inputs = { "input": context.sample.input, "prediction": context.output }
            if "retrieved_docs" in context.artifacts:
                inputs["retrieved_docs"] = context.artifacts["retrieved_docs"]

            metadata = context.meta
            metadata.update(report.errors)
            if 'retrieval_meta' in context.artifacts:
                metadata.update(context.artifacts['retrieval_meta'])
            metadata = { k: str(metadata[k])  for k in metadata }

            run_id = uuid4()
            self.client.create_run(
                id = run_id,
                name=context.sample.sample_id,
                project_name=self.project_name,
                run_type="chain",
                inputs=inputs,
                outputs={ "results": report.results },
                extra={'metadata': metadata},
            )

            for key, score in report.scores.items():
                self.client.create_feedback(
                    run_id=run_id,
                    key=key,
                    score=score,
                )
        except Exception:
            if self.logger:
                self.logger.exception("LangSmith upload failed")
            if self.strict:
                raise

def build_client(config):
    return Client(
        api_url = config.endpoint,
        api_key = config.api_key,
        timeout_ms = config.timeout_s * 1000, 
        hide_metadata = None
    )

def build_reporter(config):
    if config.type == 'langsmith':
        client = build_client(config)
        return LangSmithReporter(client, project_name=config.project_name)

    raise ValueError(f'{config.type}은 지원되지 않습니다.')