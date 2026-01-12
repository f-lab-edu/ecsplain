from typing import Annotated, List, Literal, Union

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LangSmithReporterConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LANGSMITH_",
        extra="ignore",
    )

    type: Literal["langsmith"]

    enabled: bool = False
    strict: bool = False  # True면 업로드 실패 시 eval 실패 처리

    # LangSmith 표준 환경변수 이름과 맞추고 싶으면 그대로 쓰고,
    # 이미 프로젝트에서 prefix 정책이 있으면 여기서 흡수하면 됩니다.
    api_key: str = ""
    endpoint: str = "https://api.smith.langchain.com"  # 필요 시 변경
    project_prefix: str = "rag-eval"  
    project_name: str = ""                 # 최종 project name의 prefix
    tags: List[str] = Field(default_factory=list)

    # 운영 옵션
    timeout_s: int = 20
    max_retries: int = 3
    batch_size: int = 50

ReporterConfigUnion = Annotated[
    Union[LangSmithReporterConfig],
    Field(discriminator="type")
]    