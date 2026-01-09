import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from preprocess.retrieval.config import get_config
from preprocess.retrieval.storage import make_storage


def prepare_test(config, data_path, pool_path, vector_path):
    config.chroma_dir = vector_path
    docs = load_docs(config, data_path)
    build_vectorstore(config, docs, pool_path) 

    return docs

def load_docs(config, data_path):
    docs = list()
    with data_path.open("r", encoding="utf-8") as read_fp:
        for line in read_fp:
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)
            doc = Document(
                page_content = js['source'],
                meta_data = {
                  k: js[k] for k in js if k != 'source'
                }
            )
            docs.append(doc)

    return docs

def build_vectorstore(config, docs, pool_path):
    embeddings = OpenAIEmbeddings(
        model = config.embedding_model, 
        api_key = config.openai_api_key
    )
    Chroma.from_documents(
        documents = docs,
        embedding = embeddings,
        persist_directory = config.chroma_dir
    )

def bytes_to_docs(byte_seq):
    data = byte_seq.decode('utf-8')
    docs = list()
    for line_id, line in enumerate(data.split('\n')):
        if not line: 
            continue
        js = json.loads(line)
        doc = Document(
            page_content = js['source'],
            meta_data = {
                k: js[k] for k in js if k != 'source'
            }
        )
        docs.append(doc)
    
    return docs

def build_service_config(eval_config):
    from core.expl.config import config
    # config.openai_api_key = eval_config.generation_model_config.model.api_key
    # config.openai_base_url = eval_config.generation_model_config.model.base_url

    # config.openai_model = eval_config.generation_model_config.model.name
    # config.embedding_model = eval_config.retrieval_model_config.model.name

    # config.retrieval_k = eval_config.retrieval_model_config.runtime.top_k
    # config.temperature = eval_config.generation_model_config.runtime.temperature 
    # config.reasoning_effort = eval_config.generation_model_config.runtime.reasoning_effort 

    config.chroma_dir = eval_config.retrieval_model_config.runtime.embedding_path
    config.data_dir = Path('./tests/data/chroma_db')

    return config

def load_data_from_s3(name, date_str, run_id):
    config = get_config('aws_s3')

    storage = make_storage(config)  
    eval_data_key = storage.full_key(
            name, date_str, run_id
    )
    eval_data = storage.load(eval_data_key)
    eval_data = bytes_to_docs(eval_data)

    return eval_data[:5]

def download_vectorstore_from_s3(config):
    dataset_config = config.dataset_config 
    pool_config = config.retrieval_pool_config

    storage_config = get_config('aws_s3')
    storage = make_storage(storage_config)

    dataset_id = dataset_config.id
    eval_pool_prefix = storage.full_key(
        dataset_id.namespace, dataset_id.dt, dataset_id.run_id
    )
    eval_pool = storage.download_prefix(
        eval_pool_prefix, pool_config.vectorstore.persist_dir
    )

    return eval_pool