from pydantic_settings import BaseSettings
from preprocess.retrieval.config import LocalStorageConfig, AWSS3Config, MultiStorageConfig
from preprocess.retrieval.storage import LocalStorage, S3Storage
from preprocess.retrieval.utils import build_retrieval_documents, build_retrieval_metadata

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def make_document_builder(config: BaseSettings = None):
    if config.document_type == 'retrieval':
        return build_retrieval_metadata, build_retrieval_documents 
    
    raise ValueError(f'{config.document_type}은 지원되지 않습니다.')

def make_splitter(config: BaseSettings):
    if config.splitter_type == 'recursive': 
        return RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.overlap_size,
        )
    
    raise ValueError(f'{config.splitter_type}은 지원되지 않습니다.')

def make_embeddings(config: BaseSettings):
    if config.embedding_type == 'openai':
        return OpenAIEmbeddings(
            api_key=config.openai_api_key,
            model=config.emb_model,
            chunk_size = 200
        )
    
    raise ValueError(f'{config.embedding_type}은 지원되지 않습니다.')

def make_vectorstore_builder(config: BaseSettings):
    if config.vectorstore_type == 'chroma_doc':
        return Chroma.from_documents

    raise ValueError(f'{config.vectorstore_type}은 지원되지 않습니다.')