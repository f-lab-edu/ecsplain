import argparse
from bs4 import BeautifulSoup 
import dateutil
from datetime import datetime
import json 
import os 
import openai
import requests
import sys 
import time 
import urllib
from urllib.parse import urlparse
import uuid
from pathlib import Path

from preprocess.retrieval.config import get_config
from preprocess.retrieval.crawl.service import NaverNewsService
from preprocess.retrieval.storage import LocalStorage, S3Storage


def main(service_config, local_config, s3_config):
    local_storage = LocalStorage(local_config)
    s3_storage = S3Storage(s3_config)

    data = local_storage.load(service_config.read_file_path, 'jsonl')
    n_service = NaverNewsService(service_config)

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex

    base_path = service_config.write_directory_path / date_str / run_id
    eval_path = base_path / 'retrieval_augmented_newses.jsonl'
    pool_path = base_path / 'retrieval_pool.jsonl'
    for i, js in enumerate(data): 
        js, newses = n_service.fetch_news(js)

        local_storage.save(
            eval_path, [js], 'jsonl', 
            mode = 'a'
        )
        local_storage.save(
            pool_path, [ news.to_dict() for news in newses ], 'jsonl', 
            mode = 'a'
        )

        break

    try:
        news_s3_key = s3_storage.full_key(
            'retrieval_augmented_newses', date_str, run_id
        )
        content = eval_path.read_bytes()  
        news_uri = s3_storage.save(
            news_s3_key, content, s3_config.content_type
        )

        pool_s3_key = s3_storage.full_key(
            'retrieval_pool', date_str, run_id
        )
        content = pool_path.read_bytes() 
        pool_uri = s3_storage.save(
            pool_s3_key, content, s3_config.content_type
        )

        import pdb;pdb.set_trace()
        meta_key = local_storage.full_key('meta', date_str, run_id, 'json')
        meta_data = {
            'date_str': date_str, 'run_id': run_id,
            'news_s3': {'key': str(news_s3_key), 'uri': news_uri},
            'pool_s3': {'key': str(pool_s3_key), 'uri': pool_uri} 
        }
        local_storage.save(
            meta_key, meta_data, 'json', 
            mode = 'w'
        )
    except Exception as e: 
        print(f"S3에 저장 실패")
        if s3_storage.exists(news_s3_key):
            print(f"❌ S3에서 {news_s3_key} 롤백")
            s3_storage.delete(news_s3_key)
        if s3_storage.exists(pool_s3_key):
            print(f"❌ S3에서 {pool_s3_key} 롤백")
            s3_storage.delete(pool_s3_key)
    
if __name__ == '__main__':
    import pdb;pdb.set_trace()
    service_config = get_config('news_service')

    local_config = get_config('local_storage')
    s3_config = get_config('aws_s3')


    main(service_config, local_config, s3_config)

