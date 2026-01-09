from pathlib import Path

import yaml
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from eval.types import VectorStoreType

BASE_DIR = Path(__file__).resolve().parents[3]


def load_pool_infos(info_path):
    with open(info_path, 'r') as info_fp: 
        infos = yaml.safe_load(info_fp)
    
    return infos

def build_embeddings(config):
    if config.kind == 'commercial':
        if config.provider == 'openai':
            return OpenAIEmbeddings(
                model=config.name,
                api_key=config.api_key
            )

    raise ValueError(f'{config.provider}:{config.name}은 지원되지 않습니다.')

def build_vectorstore(config, embeddings):
    if config.vectorstore.type == VectorStoreType.chroma:
        data_path = config.vectorstore.persist_dir / f'chroma_{config.sample_type}'
        return Chroma(
            embedding_function=embeddings, 
            persist_directory=str(data_path)
        )
    
    raise ValueError(f'{config.vectorstore.type}은 지원되지 않습니다.')


def build_chroma_retriever(model_config, pool_config):
    embeddings = build_embeddings(model_config.impl)
    vectorstore = build_vectorstore(pool_config, embeddings)

    retriever = vectorstore.as_retriever(
        search_type = model_config.runtime.search_type,
        search_kwargs = {
            'k': model_config.runtime.top_k
        }
    )

    return retriever 

def load_retrievers_with_pool_infos(pool_infos, config):
    retrievers = dict()

    embeddings = OpenAIEmbeddings(
        api_key=config.openai_api_key,
    )

    for pool_name, pool_info in pool_infos.items():
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=str(
                BASE_DIR / Path(pool_info['vectorstore_dir'])
            )
        )

        retriever = vectorstore.as_retriever(
            search_type = pool_info.get('search_type', 'similarity'),
            search_kwargs = { 'k': pool_info.get('k', 5) }
        )

        retrievers[pool_name] = retriever 

    return retrievers 
