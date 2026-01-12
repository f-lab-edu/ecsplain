from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from eval.types import ComponentRuntime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class RetrievalRuntimeConfig(ComponentRuntime):

    embedding_path: str = ''

    top_k: int = 5 
    score_threshold: float = 0.
    search_type: Literal["similarity", "mmr"] = 'similarity'

class GenerationRuntimeConfig(ComponentRuntime):

    temperature: float =0.
    top_p: float = 1.
    max_tokens: int = 512
    stop: list[str] = Field(default_factory=list)
    reasoning_effort: Literal["high", "medium", "low", None] = None 

    prompt_path: Path = Field(default_factory=Path)

    @field_validator('prompt_path', mode='before')
    @classmethod
    def parse_prompt_path(cls, v):
        v = Path(v) 
        if not v.is_absolute():
            v = PROJECT_ROOT / v

        return v