import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

_DATETIME_FMT = "%Y%m%d-%H%M%S"
_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")

class DatasetId(BaseModel):
    namespace: str = Field(default="")
    dt: str = Field(default="")
    run_id: str = Field(default="")
    ext: str = Field(default="")

    @field_validator("dt")
    @classmethod
    def validate_dt(cls, v: str) -> str:
        if not v:
            return v
        datetime.strptime(v, _DATETIME_FMT)
        return v

    @field_validator("run_id")
    @classmethod
    def validate_run_id_hex32(cls, v: str) -> str:
        if not v:
            return v
        if not _UUID_HEX_RE.match(v):
            raise ValueError("run_id must be 32 lowercase hex characters (no hyphens).")
        return v

    def as_prefix(self) -> str:
        return f"{self.namespace}/{self.dt}/{self.run_id}"

    @classmethod
    def parse(cls, s: str) -> "DatasetId":
        parts = [p for p in s.split("/") if p]
        if len(parts) != 3:
            raise ValueError('expected "namespace/dt/run)id"')
        return cls(namespace=parts[0], dt=parts[1], run_id=parts[2])

class DatasetConfig(BaseSettings):
    name: str = ''
    id: DatasetId = Field(default_factory=DatasetId)
    version: str = ''
    path: Path = Path('')

    manifest_path: Path = Path('')

    @field_validator("id", mode="before")
    @classmethod
    def parse_id(cls, v):
        if isinstance(v, str):
            return DatasetId.parse(v)
        return v