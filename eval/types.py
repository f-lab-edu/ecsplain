from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, Optional, TypeVar, Union

from pydantic import BaseModel

JSONDict = Dict[str, Any]

class ComponentImpl(BaseModel):
    ...

class ComponentRuntime(BaseModel):
    ...

TImpl = TypeVar("TImpl", bound = ComponentImpl)
TRuntime = TypeVar("TRuntime", bound = ComponentRuntime)

class ComponentConfig(BaseModel, Generic[TImpl, TRuntime]):
    impl: TImpl 
    runtime: TRuntime

class VectorStoreType(str, Enum):
    chroma = "chroma"
    faiss = "faiss"
    elastic = "elastic"
    bm25 = "bm25"

class PoolSyncMode(str, Enum):
    always = "always"
    if_missing = "if_missing"
    never = "never"

@dataclass(frozen=True)
class Sample: 
    sample_id: str
    input: JSONDict 

@dataclass(frozen=True)
class RAGSample(Sample):
    ret_expected: Optional[JSONDict] = None 
    gen_expected: Optional[JSONDict] = None 
    meta: JSONDict = field(default_factory=dict)

@dataclass
class RunContext:
    sample: Sample
    output: Union[dict, None] = None
    artifacts: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

@dataclass 
class EvalContext:
    run_id: str
    sample: Sample
    artifacts: JSONDict
    output: dict
    meta: JSONDict 


@dataclass(frozen=True)
class RunOutput:
    sample: Sample 
    output: JSONDict 
    artifacts: JSONDict = field(default_factory = dict)
    meta: JSONDict = field(default_factory = dict)

    def to_eval(self) -> EvalContext: 
        return EvalContext(
            run_id = self.meta['task_run_id'],
            sample = self.sample,
            artifacts = self.artifacts, 
            output = self.output,
            meta = self.meta
        )


@dataclass
class MetricResult:
    name: str 
    score: Union[float, int, None] = None
    passed: Optional[bool] = None 
    details: JSONDict = field(default_factory=dict)
    erorr: Optional[str] = None 


@dataclass(frozen=True)
class EvalResult:
    name: str
    metric: MetricResult 
    details: Optional[Any] = None      # e.g., raw judge response, rationale
    artifacts: Dict[str, Any] = field(default_factory=dict)  # e.g., prompt, token_usage, latency_ms


@dataclass(frozen=True)
class EvalReport:
    """
    Aggregated output for one sample evaluation.
    """
    run_id: str
    sample_id: Optional[str]
    # config_fingerprint: str

    results: Dict[str, EvalResult]
    scores:  Dict[str, Union[int, float, None]]
    ok: bool
    errors: Dict[str, str] = field(default_factory=dict)

    # handy rollups (optional)
    score_mean: Optional[float] = None

