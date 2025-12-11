from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field
from dotenv import load_dotenv
from typing import List
from functools import lru_cache

from preprocess.retrieval.utils import get_latest_datetime_dir, get_latest_uuid_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class QgenConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QUERY_GEN_", 
        extra="ignore",
    )

    openai_api_key: str 
    openai_base_url: str 

    # Model Setting
    openai_model: str 

    # Inference Setting 
    temperature: float
    reasoning_effort: str

    prompt_path: str

    @field_validator('prompt_path')
    @classmethod
    def resolve_prompt(cls, v: str) -> Path: 
        v = Path(v)
        if not v.is_absolute():
            v = (PROJECT_ROOT / v).resolve() 
        return v

class NaverAPIConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NAVER_API_",
        extra="ignore",
    )

    llm_config: QgenConfig

    base_url: str
    category: str
    display_num: int 

    client_id: str
    client_secret: str

    article_num: int

class NaverNewsCrawlConfig(BaseSettings): 
    model_config = SettingsConfigDict(
        env_prefix="NAVER_NEWS_CRAWL_",
        extra="ignore",
    )

    err_cnt_thr: int 

    max_proxy_num: int 
    proxies: List[str]

    min_delay: float 
    max_delay: float

    article_num: int

    @field_validator('proxies', mode='before')
    @classmethod 
    def parse_proxies(cls, v): 
        if not v: 
            return list() 
        if isinstance(v, str):
            return [ p.strip() for p in v.split(',') if p.strip()]
        if isinstance(v, list):
            return [ p.strip() for p in v if p.strip()]

        raise TypeError(f"주어진 파싱 타입 {type(v)}은 파싱을 지원하지 않습니다.") 

class NaverNewsServiceConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NAVER_NEWS_SERVICE_",
        extra="ignore"
    )

    read_directory_path:str 
    write_directory_path:str

    read_file_path:str 
    write_file_path:str

    api_config: NaverAPIConfig
    crawl_config: NaverNewsCrawlConfig

    @field_validator('read_directory_path')
    @classmethod
    def parse_read_directory_path(cls, v):
        v = Path(v)
        if not v.is_absolute():
            v = (PROJECT_ROOT / v).resolve() 
        return v
    
    @field_validator('write_directory_path')
    @classmethod 
    def parse_write_directory_path(cls, v):
        v = Path(v)
        if not v.is_absolute():
            v = (PROJECT_ROOT / v).resolve() 
        return v
    
    @field_validator('read_file_path')
    @classmethod 
    def parse_read_file_path(cls, v): 
        v = Path(v)
        if not v.is_absolute():
            v = (PROJECT_ROOT / v)
        return v 
    
    @field_validator('write_file_path')
    @classmethod 
    def parse_write_file_path(cls, v):
        v = Path(v)
        if not v.is_absolute():
            v = (PROJECT_ROOT / v)
        return v 


class StorageConfig(BaseSettings):
    ...

class LocalStorageConfig(StorageConfig):
    model_config = SettingsConfigDict(
        env_prefix="LOCAL_",
        extra="ignore"
    )

    base_path: str
    storage_type: str = 'local_storage'

    @field_validator("base_path")
    @classmethod
    def parse_base_path(cls, v): 
        v = Path(v)
        if not v.is_absolute():
            v = (PROJECT_ROOT) / v

        return v 

class AWSConnectionConfig(BaseSettings): 
    model_config = SettingsConfigDict(
        env_prefix="AWS_CONN_",
        extra="ignore"
    )

    access_key_id: str 
    secret_access_key: str 
    default_region: str

class AWSS3Config(StorageConfig):
    model_config = SettingsConfigDict(
        env_prefix="AWS_S3_",
        extra="ignore"
    )

    conn_config: AWSConnectionConfig

    service_type: str = 's3'
    endpoint_url: str

    bucket: str
    retrieval_prefix: str 

    ext: str = 'jsonl'
    content_type: str = 'application/x-ndjson'

class MultiStorageConfig(StorageConfig):
    model_config = SettingsConfigDict(
        env_prefix="MULTI_STORAGE_",
        extra="ignore"
    )

    storage_types: List[str]
    storage_configs: List[StorageConfig] = Field(default_factory=list)

    @field_validator('storage_types')
    @classmethod
    def parse_storage_types(cls, v):
        if isinstance(v, str):
            return [ p.strip() for p in v.split(",") ]
        elif isinstance(v, list):
            return v 
        
        raise ValueError(f'{type(v)}는 지원되지 않습니다.')
    
    def model_post_init(self, __context):
        self.storage_configs = [
           get_config(storage_type) for storage_type in self.storage_types
        ]
        
class IngestConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INGEST_",
        extra="ignore"
    )

    data_dir: str 
    chunk_size: int 
    overlap_size: int 
    emb_model: str 
    openai_api_key: str
    emb_dir: str 
    retrieval_pool_type: str

    load_storage_type: str 
    save_storage_type: str
    document_type: str 
    embedding_type: str 
    splitter_type: str 
    vectorstore_type: str 

    load_storage_config: StorageConfig = Field(default_factory=StorageConfig)
    save_storage_config: StorageConfig = Field(default_factory=StorageConfig)

    def model_post_init(self, __context):
        root_dir = PROJECT_ROOT / self.data_dir / self.retrieval_pool_type
        latest_date_dir = get_latest_datetime_dir(root_dir)
        latest_uuid_dir = get_latest_uuid_dir(root_dir / latest_date_dir)
        
        self.data_dir = root_dir / latest_date_dir / latest_uuid_dir
        self.emb_dir = PROJECT_ROOT / self.emb_dir / latest_date_dir / latest_uuid_dir

        self.load_storage_config = get_config(self.load_storage_type)
        self.save_storage_config = get_config(self.save_storage_type)


@lru_cache(maxsize=1)
def get_qgen_config() -> QgenConfig: 
    return QgenConfig()

@lru_cache(maxsize=1)
def get_naver_api_config() -> NaverAPIConfig:
    return NaverAPIConfig(
        llm_config=get_qgen_config()
    ) 

@lru_cache(maxsize=1)
def get_naver_news_crawl_config() -> NaverNewsCrawlConfig:
    return NaverNewsCrawlConfig() 

@lru_cache(maxsize=1)
def get_naver_news_service_config() -> NaverNewsServiceConfig:
    return NaverNewsServiceConfig(
        api_config = get_naver_api_config(),
        crawl_config = get_naver_news_crawl_config()
    ) 

@lru_cache(maxsize=1)
def get_aws_conn_config() -> AWSConnectionConfig:
    return AWSConnectionConfig() 

def get_aws_s3_config() -> AWSS3Config:
    return AWSS3Config(
        conn_config = get_aws_conn_config()
    )

@lru_cache(maxsize=1)
def get_ingest_config() -> IngestConfig: 
    return IngestConfig()

def get_local_storage_config() -> LocalStorageConfig:
    return LocalStorageConfig()

def get_multi_storage_config() -> MultiStorageConfig: 
    return MultiStorageConfig()

def get_config(config_type) -> BaseSettings: 
    if config_type == 'query_generate':
        return get_qgen_config()
    elif config_type == 'naver_api':
        return get_naver_api_config()
    elif config_type == 'naver_news_crawl':
        return get_naver_news_crawl_config() 
    elif config_type == 'news_service':
        return get_naver_news_service_config()
    elif config_type == 'local_storage':
        return get_local_storage_config()
    elif config_type == 'aws_s3':
        return get_aws_s3_config()
    elif config_type == 'multi_storage':
        return get_multi_storage_config()
    elif config_type == 'ingest':
        return get_ingest_config()

    raise ValueError(f'Configuration type {config_type}은 지원되지 않습니다.')
