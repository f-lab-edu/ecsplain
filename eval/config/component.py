from pydantic import Field

from eval.config.model import CommercialModelConfig, ModelUnion
from eval.config.runtime import GenerationRuntimeConfig, RetrievalRuntimeConfig
from eval.types import ComponentConfig


class RetrievalComponentConfig(ComponentConfig[ModelUnion, RetrievalRuntimeConfig]):
    impl: ModelUnion = Field(default_factory=lambda: CommercialModelConfig(
        provider="openai",
        name="text-embedding-3-large",
        comment="retrieval embedder",
    ))
    runtime: RetrievalRuntimeConfig = Field(default_factory=RetrievalRuntimeConfig)
    
class GenerationComponentConfig(ComponentConfig[ModelUnion, GenerationRuntimeConfig]):
    impl: ModelUnion = Field(default_factory=lambda: CommercialModelConfig(
        provider="openai",
        name="gpt-5.1",
        comment="generation",
    ))
    runtime: GenerationRuntimeConfig = Field(default_factory=GenerationRuntimeConfig)