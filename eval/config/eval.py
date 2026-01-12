from functools import lru_cache
from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from eval.config.component import GenerationComponentConfig
from eval.config.reporter import ReporterConfigUnion
from eval.config.task import RAGTaskConfig


class ExperimentMetaInfo(BaseModel):
    title: str = ""  
    id: str = ""
    goal: str = ""
    hypothesis: str = "" 
    notes: str = ""

    git_commit_hash: str = ""

class LLMEvalConfig(BaseModel):
    type: Literal["llm_eval"]
    name: str
    gen_component_config: GenerationComponentConfig
    max_retries: int

EvaluatorConfigUnion = Annotated[
    Union[LLMEvalConfig],
    Field(discriminator="type")
]

class EvalEngineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix = 'EVAL_ENGINE__',
        env_nested_delimiter = '__',
        extra = "ignore"
    )

    strict: bool = False 

    evaluators: List[EvaluatorConfigUnion] = Field(default_factory=list)
    reporters: List[ReporterConfigUnion] = Field(default_factory=list)
    meta_info: ExperimentMetaInfo = Field(default_factory=ExperimentMetaInfo)

@lru_cache(maxsize=1)
def get_eval_config(config_type):
    if config_type == 'rag':
        return RAGTaskConfig()
    elif config_type == 'eval_engine':
        return EvalEngineConfig()
    
    raise ValueError(f'Configuraiton type {config_type}은 지원되지 않습니다.')