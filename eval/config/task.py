from typing import Annotated, Union

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from eval.config.component import GenerationComponentConfig, RetrievalComponentConfig
from eval.config.data import DatasetConfig
from eval.config.pool import RetrievalPoolConfig


class RAGTaskConfig(BaseSettings):
    model_config = SettingsConfigDict( 
        env_prefix="RAG_TASK__", 
        env_nested_delimiter="__", 
        extra="ignore", 
    )

    storage_type: str

    dataset_config: DatasetConfig = Field(default_factory=DatasetConfig)

    retrieval_pool_config: RetrievalPoolConfig = Field(default_factory=RetrievalPoolConfig)
    retrieval_model_config: RetrievalComponentConfig = Field(
        default_factory=RetrievalComponentConfig
    )

    generation_model_config: GenerationComponentConfig = Field(
        default_factory=GenerationComponentConfig
    )

    seed: int = 0

TaskUnion = Annotated[
    Union[RAGTaskConfig],
    Field(discriminator="kind")
]