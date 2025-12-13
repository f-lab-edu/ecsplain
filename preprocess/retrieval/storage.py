from abc import ABC, abstractmethod 
import json 
import os 
from pathlib import Path 
import posixpath
from pydantic_settings import BaseSettings
from preprocess.retrieval.crawl.clients import get_client
from preprocess.retrieval.config import LocalStorageConfig, AWSS3Config, MultiStorageConfig
from typing import Iterable, Dict, Optional
from botocore.exceptions import ClientError

class StorageError(Exception):
    pass

class Storage(ABC): 
    def full_key(self, key:str, date_str:str, run_id:str):
        raise NotImplementedError

    def save(self, key:str, content:str, content_type:str):
        raise NotImplementedError 
    
    def load(self, key:str, content:str, content_type:str):
        raise NotImplementedError 
    
    def delete(self, key:str) -> None:
        raise NotImplementedError 

    def exists(self, key:str) -> bool:
        raise NotImplementedError 

class LocalStorage:
    def __init__(self, config):
        self.config = config

    def full_key(self, key:str, date_str, run_id, ext=None):
        base_path =  self.config.base_path / Path(date_str) / Path(f'{run_id}')

        if ext is None:
            return base_path / Path(f'{key}')
        
        return base_path / Path(f'{key}.{ext}')

    def save(self, key:str, content:str, content_type:str, mode:str = 'w'): 
        if content_type == 'jsonl': 
            file_path = Path(key)
            os.makedirs(file_path.parent, exist_ok=True)
            self.save_jsonl(key, content, mode=mode)
        elif content_type == 'json':
            file_path = Path(key)
            os.makedirs(file_path.parent, exist_ok=True)
            self.save_json(key, content, mode=mode)
        else:
            raise TypeError(f'Content Type {content_type}은 지원되지 않습니다.')
    
    def load(self, key:str, content_type:str):
        if content_type == 'jsonl':
            return self.load_jsonl(key)
        elif content_type == 'json':
            return self.load_json(key)

        raise TypeError(f'Content Type {content_type}은 지원되지 않습니다.')
    
    def load_dir(self, key:str):
        data = dict()
        dir_path = Path(key)
        for f in dir_path.iterdir():
            if not f.is_file(): continue
            ext = f.suffix.replace('.', '')
            data[f.name] = self.load(f, ext)
        
        return data

    def get_jsonl_text(self, data): 
        return '\n'.join([
            json.dumps(js, ensure_ascii=False) for js in data 
        ])

    def save_json(self, file_path, data, mode = 'w'):
        with open(file_path, mode, encoding='utf-8') as write_fp: 
            json.dump(data, write_fp)
    
    def save_jsonl(self, file_path, data, mode = 'w'):
        with open(file_path, mode, encoding='utf-8') as write_fp: 
            write_fp.write(self.get_jsonl_text(data))
    
    def load_json(self, file_path):
        with open(file_path, 'r') as read_fp: 
            data = json.load(read_fp)
        
        return data 

    def load_jsonl(self, file_path):
        with open(file_path, 'r') as read_fp: 
            data = list() 
            for line in read_fp.readlines():
                js = json.loads(line)
                data.append(js)
        
        return data 

    def read_directory_jsonls(self):
        data = list()
        file_names = os.listdir(self.config.read_directory_path)
        for file_name in file_names: 
            file_path = Path(os.path.join(
                self.config.read_directory_path, file_name
            ))
            if not file_path.is_file(): 
                continue

            with open(file_path, 'r', encoding='utf-8') as read_fp: 
                for line in read_fp.readlines():
                    js = json.loads(line)
                    data.append(js)
        
        return data 

class S3Storage(Storage): 
    def __init__(self, config):
        self.config = config 
        self.client = get_client(config)

    def full_key(self, key, date_str, run_id, ext=None) -> str: 
        if ext is None: 
            return Path(key) / Path(date_str) / Path(f'{run_id}')

        return Path(key) / Path(date_str) / Path(f'{run_id}.{ext}')

    def save(self, key: str, content: bytes, content_type: str) -> str:
        extra = {}
        if content_type:
            extra["ContentType"] = content_type

        try:
            self.client.put_object(
                Bucket=self.config.bucket,
                Key=str(key),
                Body=content,
                **extra,
            )
        except ClientError as e:
            raise StorageError(f"S3 put_object 실패: {e}") from e

        return f"s3://{self.config.bucket}/{key}"
    
    def save_batch(self, key: str, contents: bytes, content_type: Optional[str] = None) -> str: 
        for content in contents: 
            save(key, content, content_type)
    
    def load(self, key: str) -> bytes:
        try:
            resp = self.client.get_object(Bucket=self.config.bucket, Key=str(key))
            return resp["Body"].read()
        except ClientError as e:
            raise StorageError(f"S3 get_object 실패: {e}") from e
    
    def delete(self, key: str) -> None:
        try:
            self.client.delete_object(Bucket=self.config.bucket, Key=str(key))
        except ClientError as e:
            raise StorageError(f"S3 delete_object 실패: {e}") from e

    def exists(self, key: str) -> bool:
        """객체 존재 여부."""
        try:
            self.client.head_object(Bucket=self.config.bucket, Key=str(key))
            return True
        except ClientError as e:
            # 404만 False, 나머지는 에러로
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise StorageError(f"S3 head_object 실패: bucket={self.config.bucket}, key={key}, error={e}") from e

class MultiStorage(Storage):
    def __init__(self, config):
        self.config = config 
        self._storages = [
          make_storage(storage_config)  for storage_config in self.config.storage_configs
        ]
    
    def full_key(self, key:str, date_str:str, run_id:str):
        return [
           storage.full_key(key, date_str, run_id) for storage in self._storages
        ]
    
    def save(self, keys: str, content: bytes, content_type: str):
        for key, storage in zip(keys, self._storages): 
            storage.save(key, content, content_type)
    
    def load(self, key: str, content_type: str):
        loaded = list() 
        for storage in self._storages: 
            loaded += storage.load(key, content_type) 
        
        return loaded
    

def make_storage(config: BaseSettings = None):
    if isinstance(config, LocalStorageConfig):
        return LocalStorage(config)
    elif isinstance(config, AWSS3Config):
        return S3Storage(config)
    elif isinstance(config, MultiStorageConfig):
        return MultiStorage(config) 

    raise ValueError(f'{config}은 지원되지 않습니다.')

    
