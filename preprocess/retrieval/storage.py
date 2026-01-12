import json
import os
from abc import ABC
from pathlib import Path
from typing import Iterable, List, Optional

from botocore.exceptions import ClientError
from pydantic_settings import BaseSettings

from preprocess.retrieval.config import AWSS3Config, LocalStorageConfig, MultiStorageConfig
from preprocess.retrieval.crawl.clients import get_client


class StorageError(Exception):
    pass

class Storage(ABC): 
    def full_key(self, key:str, date_str:str, run_id:str):
        raise NotImplementedError

    def save(self, key:str, content:bytes, content_type:str):
        raise NotImplementedError 
    
    def load(self, key:str, content_type:str):
        raise NotImplementedError 
    
    def delete(self, key:str) -> None:
        raise NotImplementedError 

    def exists(self, key:str) -> bool:
        raise NotImplementedError 

class LocalStorage(Storage):
    def __init__(self, config):
        self.config = config

    def full_key(self, key:str, date_str, run_id, ext=None):
        base_path =  self.config.base_path / Path(date_str) / Path(f'{run_id}')

        if ext is None:
            return base_path / Path(f'{key}')
        
        return base_path / Path(f'{key}.{ext}')

    def save(self, key:str, content:bytes, content_type:str, mode:str = 'w'): 
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
    
    def load(self, key:str, content_type:str='jsonl'):
        if content_type == 'jsonl':
            return self.load_jsonl(key)
        elif content_type == 'json':
            return self.load_json(key)

        raise TypeError(f'Content Type {content_type}은 지원되지 않습니다.')
    
    def load_dir(self, key:str):
        data = dict()
        dir_path = Path(key)
        for f in dir_path.iterdir():
            if not f.is_file(): 
                continue
            ext = f.suffix.replace('.', '')
            data[f.name] = self.load(str(f), ext)
        
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
    def __init__(self, config: AWSS3Config):
        self.config = config
        self.client = get_client(config)

    # ---------- key helpers ----------
    def _ensure_str_key(self, key: Path) -> str:
        return str(key).lstrip("/")

    def _with_ext(self, key: str) -> str:
        """논리 key에 기본 확장자를 붙여 실제 S3 object key로 만든다."""
        key = self._ensure_str_key(Path(key))
        if not self.config.ext:
            return key
        if key.endswith(f".{self.config.ext}"):
            return key
        return f"{key}.{self.config.ext}"

    def full_key(self, key: str, date_str: str, run_id: str, ext: Optional[str] = None) -> str:
        base = Path(key) / date_str / run_id
        if ext is None:
            return str(base)
        return str(base.with_suffix(f".{ext}"))

    # ---------- single object ops ----------
    def save(self, key: str, content: bytes, content_type: Optional[str] = None) -> str:
        extra = {}
        if content_type:
            extra["ContentType"] = content_type

        obj_key = self._with_ext(key)
        try:
            self.client.put_object(
                Bucket=self.config.bucket,
                Key=obj_key,
                Body=content,
                **extra,
            )
        except ClientError as e:
            error_msg = (
                "S3 put_object 실패: bucket={self.config.bucket}",
                ", key={obj_key}",
                ", error={e}"
            ) 
            raise StorageError(error_msg) from e

        return f"s3://{self.config.bucket}/{obj_key}"

    def save_batch(
        self, key_prefix: str, contents: Iterable[bytes], content_type: Optional[str] = None
    ) -> List[str]:
        """배치는 보통 key를 각 content별로 다르게 가져가야 하므로 prefix+index 전략을 씀."""
        uris: List[str] = []
        for i, content in enumerate(contents):
            uris.append(self.save(f"{key_prefix}/{i}", content, content_type))
        return uris

    def load(self, key: str, content_type: str) -> bytes:
        obj_key = self._with_ext(key)
        try:
            resp = self.client.get_object(Bucket=self.config.bucket, Key=obj_key)
            return resp["Body"].read()
        except ClientError as e:
            error_msg = (
                f"S3 get_object 실패: bucket={self.config.bucket}"
                f", key={obj_key}"
                f", error={e}"
            )
            raise StorageError(error_msg) from e

    def delete(self, key: str) -> None:
        obj_key = self._with_ext(key)
        try:
            self.client.delete_object(Bucket=self.config.bucket, Key=obj_key)
        except ClientError as e:
            error_msg = (
                f"S3 delete_object 실패: bucket={self.config.bucket}"
                f", key={obj_key}"
                f", error={e}"
            )

            raise StorageError(error_msg) from e

    def exists(self, key: str) -> bool:
        obj_key = self._with_ext(key)
        try:
            self.client.head_object(Bucket=self.config.bucket, Key=obj_key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            error_msg = (
                f"S3 head_object 실패: bucket={self.config.bucket}"
                f", key={obj_key}"
                f", error={e}"
            )
            raise StorageError(error_msg) from e

    def list_keys(self, prefix: str, max_keys: Optional[int] = None) -> List[str]:
        """prefix 아래 객체 key들을 모두 나열. (S3 '폴더' 조회의 기본)"""
        prefix = self._ensure_str_key(Path(prefix)).rstrip("/") + "/"
        paginator = self.client.get_paginator("list_objects_v2")

        out: List[str] = []
        try:
            for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    k = obj["Key"]
                    if k.endswith("/"):
                        continue
                    out.append(k)
                    if max_keys is not None and len(out) >= max_keys:
                        return out
            return out
        except ClientError as e:
            error_msg = (
                f"S3 list_objects_v2 실패: bucket={self.config.bucket}"
                f", prefix={prefix}"
                f", error={e}"
            )
            raise StorageError(error_msg) from e

    def is_directory(self, key: str) -> bool:
        """key를 '폴더(prefix)'로 간주할 수 있는지: key/ 아래에 객체가 1개라도 있으면 True."""
        keys = self.list_keys(key, max_keys=1)
        return len(keys) > 0

    def download_prefix(self, s3_prefix: str, local_dir: Path) -> List[Path]:
        """
        s3://bucket/s3_prefix 아래를 로컬로 구조 그대로 다운로드.
        local_dir 아래에 prefix 하위 상대경로로 저장.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        s3_prefix = self._ensure_str_key(Path(s3_prefix)).rstrip("/") + "/"
        keys = self.list_keys(s3_prefix)

        downloaded: List[Path] = []
        for key in keys:
            rel = Path(key).relative_to(s3_prefix)
            dst = local_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.client.download_file(self.config.bucket, key, str(dst))
            except ClientError as e:
                error_msg = (
                    f"S3 download_file 실패: bucket={self.config.bucket}"
                    f", key={key}"
                    f", dst={dst}"
                    f", error={e}"
                )
                raise StorageError(error_msg) from e
            downloaded.append(dst)

        return downloaded

class MultiStorage:
    def __init__(self, config):
        self.config = config 
        self._storages = [
          make_storage(storage_config)  for storage_config in self.config.storage_configs
        ]
    
    def full_key(self, key:str, date_str:str, run_id:str):
        return [
           storage.full_key(key, date_str, run_id) for storage in self._storages
        ]
    
    def save(self, keys: List[str], content: bytes, content_type: str):
        for key, storage in zip(keys, self._storages): 
            storage.save(key, content, content_type)
    
    def load(self, key: str, content_type: str):
        loaded = list() 
        for storage in self._storages: 
            loaded += storage.load(key, content_type) 
        
        return loaded
    

def make_storage(config: BaseSettings):
    if isinstance(config, LocalStorageConfig):
        return LocalStorage(config)
    elif isinstance(config, AWSS3Config):
        return S3Storage(config)
    elif isinstance(config, MultiStorageConfig):
        return MultiStorage(config) 

    raise ValueError(f'{config}은 지원되지 않습니다.')

    
