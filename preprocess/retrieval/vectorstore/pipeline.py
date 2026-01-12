from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol, Union

from langchain_openai import OpenAIEmbeddings

from preprocess.retrieval.config import IngestConfig
from preprocess.retrieval.storage import make_storage
from preprocess.retrieval.vectorstore.factory import (
    make_document_builder,
    make_embeddings,
    make_splitter,
    make_vectorstore_builder,
)


@dataclass 
class IngestContext:
    config: IngestConfig 

    raw_data: Dict[str, Any] = dict() 
    docs: Dict[str, List[Any]] = dict()
    embeddings: Union[OpenAIEmbeddings, None] = None
    vectorstores: Dict[str, Any] = dict()

    meta_data: Dict[str, Any] = dict()

class Step(Protocol):
    def run(self, ctx: IngestContext) -> IngestContext:
        ...

class LoadRawDataStep(Step):
    def run(self, ctx: IngestContext) -> IngestContext:
        storage = make_storage(ctx.config.load_storage_config)
        loaded = storage.load_dir(ctx.config.data_dir)

        if not loaded:
            raise RuntimeError(f"[LoadRawDataStep] 데이터가 없습니다.: {ctx.config.data_dir}")

        ctx.raw_data = dict() 
        for k in loaded: 
            if 'meta' in k: 
                ctx.meta_data = loaded[k]
            elif 'pool' in k:
                ctx.raw_data[k] = loaded[k]

        return ctx 

class SplitStep(Step):
    def run(self, ctx: IngestContext) -> IngestContext:
        if ctx.raw_data is None: 
            raise RuntimeError(
                "[SplitStep] raw_data가 비어있습니다. LoadRawDataStep을 먼저 실행해야 합니다"
            )

        splitter = make_splitter(ctx.config)
        metadata_builder, document_builder = make_document_builder(ctx.config)
        docs: Dict[str, List[Any]] = dict() 

        for raw_k in ctx.raw_data:
            docs[raw_k] = list()
            for orig_data in ctx.raw_data[raw_k]:
                content_k = 'source' if 'source' in orig_data else 'content'
                fragments = splitter.split_text(orig_data[content_k])
                for frag_id, frag in enumerate(fragments):
                    metadata = {
                      k: orig_data[k]  
                      for k in orig_data if k != content_k
                    }
                    metadata['id'] += f'.{frag_id}'
                    doc = document_builder(frag, metadata)
                    docs[raw_k].append(doc)

        ctx.docs = docs 
        return ctx 

class BuildVectorStoreStep(Step):
    def run(self, ctx: IngestContext) -> IngestContext:
        if ctx.docs is None: 
            raise RuntimeError(
                "[BuildVectorStoreStep] docs가 비어있습니다. SplitStep 부터 실행해야 합니다."
            )

        vectorstores = dict() 
        embeddings = make_embeddings(ctx.config)
        vectorstore_builder = make_vectorstore_builder(ctx.config)
        for k in ctx.docs:
            vectorstores[k] = vectorstore_builder(
                documents = ctx.docs[k], 
                embedding = embeddings,
                persist_directory = str(ctx.config.emb_dir)
            )

        ctx.embeddings = embeddings
        ctx.vectorstores = vectorstores

        return ctx 
    
class SaveVectorStoreStep(Step):
    def run(self, ctx: IngestContext) -> IngestContext:  
        storage = make_storage(ctx.config.save_storage_config)
        for k in ctx.vectorstores:
            name, ext = k.split('.')
            name = f'{ctx.config.retrieval_pool_type}/{name}'
            date_str = ctx.meta_data['date_str']
            run_id = ctx.meta_data['run_id']
            root_key = storage.full_key(name, date_str, run_id)

            emb_dir = Path(ctx.config.emb_dir)
            for file_path in emb_dir.rglob('*'):
                if file_path.is_file():
                    rel_path = file_path.relative_to(emb_dir)
                    content = file_path.read_bytes()
                    content_type = "application/octet-stream"
                    if isinstance(root_key, Iterable):
                        keys = [
                            f'{k}/{ctx.config.vectorstore_type}/{rel_path.as_posix()}' 
                            for k in root_key
                        ]
                        storage.save(keys, content, content_type)
                    else:
                        key = f'{root_key}/{ctx.config.vectorstore_type}/{rel_path.as_posix()}'
                        storage.save(key, content, content_type)
    
        return ctx


class IngestPipeline:

    def __init__(self, config, steps=None) -> None:
        self.ctx = IngestContext(config=config)
        if steps is None:
            self.steps = [
                LoadRawDataStep(), SplitStep(),
                BuildVectorStoreStep(), SaveVectorStoreStep()
            ]
        else:
            self.steps = steps
        
    def run(self) -> IngestContext:
        ctx = self.ctx
        for step in self.steps:
            ctx = step.run(ctx)

        return ctx  
    
    def add_step(self, step, index=None):
        if index is None:
            self.steps.append(step)
        else: 
            self.steps.insert(step, index)
    
    def remove_step(self, step, index=None):
        if index is None: 
            self.steps.pop()
        else:
            self.steps.pop(index)   
    

