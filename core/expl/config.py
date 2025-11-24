from pydantic_settings import BaseSettings, SettingsConfigDict 
from pathlib import Path 

class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_API_", 
        extra="ignore",
    )

    openai_api_key: str 
    openai_base_url: str 

    # Model Setting
    openai_model: str 
    embedding_model: str 

    # Inference Setting 
    retrieval_k: int
    temperature: float
    reasoning_effort: str

    # Data Setting 
    chroma_dir: str 
    data_dir: str

    prompt_path: str = './core/expl/prompts/expl_detailed.txt'

config = Config()
