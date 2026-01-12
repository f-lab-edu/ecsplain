import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from pydantic import Field

from core.expl.build import build_generator
from core.retrieval.vectorstore.build_chroma import build_chroma_retriever
from eval.config.task import RAGTaskConfig
from eval.interfaces import Step, Task
from eval.prompts.builder import get_prompt_builder
from eval.runner.pipeline import GenerateStep, Pipeline, PrepareMetaStep, RetrieveStep
from eval.types import RAGSample, VectorStoreType
from eval.utils import bytes_to_jsonl, make_config_hash
from preprocess.retrieval.config import AWSS3Config, StorageConfig, get_config
from preprocess.retrieval.storage import make_storage


class RAGTask(Task): 
    name: str = 'rag_task'

    config: RAGTaskConfig 
    _storage_config: Optional[StorageConfig] = None

    project_root: Path

    _storage: Optional[Any] = Field(default=None, init=False, repr=False)
    _retriever: Optional[Any] = Field(default=None, init=False, repr=False)
    _generator: Optional[Any] = Field(default=None, init=False, repr=False)
    _prompt_builder: Optional[Any] = Field(default=None, init=False, repr=False)

    def __init__(self, config):
        self.config = config 

        self.crated_at = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.id = uuid.uuid4().hex

        self._build_storage()
        self._build_retriever()
        self._build_generator()

    def effective_config(self):
        config = self.config
        effective_config = {
            'dataset_name': config.dataset_config.name,
            'dataset_namespace': config.dataset_config.id.namespace,
            'dataset_dt': config.dataset_config.id.dt,
            'dataset_run_id': config.dataset_config.id.run_id,
            'dataset_version': config.dataset_config.version,
            'vectorstore_type': config.retrieval_pool_config.vectorstore.type,
            'vectorstore_persist_dir': str(config.retrieval_pool_config.vectorstore.persist_dir),
        }

        retrieval_impl_config = self.config.retrieval_model_config.impl
        retrieval_runtime_config = self. config.retrieval_model_config.runtime 
        if retrieval_impl_config.kind == 'commercial':
            effective_config.update({
                'retrieval_kind': 'commercial', 
                'retrieval_provider': retrieval_impl_config.provider,
                'retreival_name': retrieval_impl_config.name,
                'retrieval_top_k': retrieval_runtime_config.top_k,
                'retrieval_score_threshold': retrieval_runtime_config.score_threshold,
                'retrieval_search_type': retrieval_runtime_config.search_type
            })
        elif retrieval_impl_config.kind == 'public':
            effective_config.update({
                'retrieval_kind': 'public', 
                'retrieval_name': retrieval_impl_config.name,
                'retrieval_ckt_path': str(retrieval_impl_config.checkpoint_path),
                'retrieval_top_k': retrieval_runtime_config.top_k,
                'retrieval_score_threshold': retrieval_runtime_config.score_threshold,
                'retrieval_search_type': retrieval_runtime_config.search_type 
            })

        generation_impl_config = self.config.generation_model_config.impl
        generation_runtime_config = self. config.generation_model_config.runtime 
        if generation_impl_config.kind == 'commercial':
            effective_config.update({
                'generation_kind': 'commercial', 
                'generation_provider': generation_impl_config.provider,
                'generation_name': generation_impl_config.name,
                'generation_temperature': generation_runtime_config.temperature,
                'generation_top_p': generation_runtime_config.top_p,
                'generation_max_tokens': generation_runtime_config.max_tokens,
                'generation_stop': generation_runtime_config.stop,
                'generation_reasoning_effort': generation_runtime_config.reasoning_effort,
                'generation_prompt_path': str(generation_runtime_config.prompt_path)
            })
        elif generation_impl_config.kind == 'public':
            effective_config.update({
                'generation_kind': 'public', 
                'generation_name': generation_impl_config.name,
                'generation_ckt_path': str(generation_impl_config.checkpoint_path),
                'generation_temperature': generation_runtime_config.temperature,
                'generation_top_p': generation_runtime_config.top_p,
                'generation_max_tokens': generation_runtime_config.max_tokens,
                'generation_stop': generation_runtime_config.stop,
                'generation_reasoning_effort': generation_runtime_config.reasoning_effort,
                'generation_prompt_path': str(generation_runtime_config.prompt_path)
            })
        
        return effective_config

    def make_hash(self):
        effective_config = self.effective_config() 
        config_hash = make_config_hash(effective_config)

        return config_hash

    def build_pipeline(self) -> Pipeline:
        steps: List[Step] = list() 

        retrieval_runtime = self.config.retrieval_model_config.runtime 
        generation_runtime = self.config.generation_model_config.runtime

        steps.append(PrepareMetaStep())
        steps.append(RetrieveStep(
            self.retriever, 
            retrieval_runtime.top_k, 
            retrieval_runtime.score_threshold,
            retrieval_runtime.search_type
        ))
        steps.append(GenerateStep(
            self.generator,
            generation_runtime.temperature,
            generation_runtime.top_p,
            generation_runtime.max_tokens,
            generation_runtime.stop,
            generation_runtime.reasoning_effort,
            generation_runtime.prompt_path
        ))

        return Pipeline(steps)

    @property
    def storage_config(self):
        if self._storage_config is None:
            self._storage_config = get_config(self.config.storage_type)
        
        return self._storage_config 

    @property
    def storage(self):
        if self._storage is None: 
            self._storage = self._build_storage() 
        
        return self._storage

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = self._build_retriever()
        
        return self._retriever 

    @property
    def generator(self) -> Any: 
        if self._generator is None: 
            self._generator = self._build_generator()
        
        return self._generator 

    def load_samples(self) -> List[RAGSample]:
        dataset_id = self.config.dataset_config.id 
        key = self.storage.full_key(
            dataset_id.namespace, dataset_id.dt, dataset_id.run_id, dataset_id.ext 
        )

        data = self.storage.load(key, dataset_id.ext)
        samples = list() 
        if dataset_id.ext == 'jsonl':
            data = bytes_to_jsonl(data)
            for js in data:
                samples.append(
                    RAGSample(
                        sample_id = js["id"],
                        input = js["source"],
                        ret_expected = js["retrievals"],
                        meta = js["metadata"]
                    )
                )
        else: 
            raise ValueError(f'{dataset_id.ext}은 지원되지 않습니다.')
        
        return samples 

    def _build_storage(self):
        self._storage = make_storage(self.storage_config)

    def download_vectorstore(self, pool_config):
        dataset_id = pool_config.dataset_config.id 
        pool_prefix = self.storage.full_key(
            dataset_id.namespace, dataset_id.dt, dataset_id.run_id
        )
        pool = self.storage.download_prefix(
            pool_prefix, pool_config.vectorstore.persist_dir
        )

        return pool

    def _build_retriever(self):
        model_config = self.config.retrieval_model_config
        pool_config = self.config.retrieval_pool_config

        download_cond = all([
          isinstance(self.storage_config, AWSS3Config),
          not self.config.retrieval_pool_config.vectorstore.persist_dir.exists()   
        ])
        if download_cond:
            self.download_vectorstore(pool_config)

        if pool_config.vectorstore.type == VectorStoreType.chroma:
            self._retriever = build_chroma_retriever(model_config, pool_config)
        else:
            raise ValueError(f'{pool_config.vectorstore.type}은 현재 지원되지 않습니다.')

    def _build_generator(self):
        gen_config = self.config.generation_model_config

        self._generator = build_generator(gen_config)

    def build_prompt_builder(self):
        gen_runtime = self.config.generation_model_config.runtime

        prompt_format = gen_runtime.prompt_path.read_text()
        prompt_builder = get_prompt_builder(gen_runtime.prompt_path, prompt_format)

        return prompt_builder
