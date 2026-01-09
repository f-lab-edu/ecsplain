from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from eval.config.data import DatasetConfig
from eval.types import VectorStoreType

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class VectorStoreConfig(BaseSettings):
    type: VectorStoreType = VectorStoreType.chroma

    distance: Literal["cosine", "l2"] = 'cosine'

class ChromaConfig(VectorStoreConfig):
    type: Literal[VectorStoreType.chroma] = VectorStoreType.chroma 

    persist_dir: Path = Path('')
    collection: str = ''

    @field_validator('persist_dir')
    @classmethod
    def parse_persist_dir(cls, v):
        v = Path(v)
        if v.is_absolute():
            return v
        
        return PROJECT_ROOT / v 

VectorStoreUnion = Annotated[
    Union[ChromaConfig],
    Field(discriminator="type"),
]

class RetrievalPoolConfig(BaseSettings):
    dataset_config: DatasetConfig
    sample_type: str

    vectorstore: VectorStoreUnion = Field(default_factory=ChromaConfig)
