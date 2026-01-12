from pathlib import Path
from typing import Dict, List, Union
from urllib.parse import urlparse

from langchain_core.documents import Document


def parse_s3_uri(uri: str):
    parsed = urlparse(uri)

    if parsed.scheme == 's3':
        data_type, date_str, run_id = parsed.path.split("/")
        run_id = run_id.split('.')[0]

        return data_type, date_str, run_id 
    
    raise ValueError(f'Storage {parsed.scheme}은 지원되지 않습니다.')

def get_latest_datetime_dir(root_dir: Path) -> Union[Path, None]:
    root = Path(root_dir)
    dt_dirs = [p for p in root.iterdir() if p.is_dir()]

    if not dt_dirs:
        return None

    latest_dt_dir = max(dt_dirs, key=lambda p: p.name)
    return latest_dt_dir

def get_latest_uuid_dir(root_dir: Path) -> Union[Path, None]:
    if root_dir is None:
        return None

    uuid_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    if not uuid_dirs:
        return None

    latest_uuid_dir = max(uuid_dirs, key=lambda p: p.stat().st_mtime)
    return latest_uuid_dir

def create_documents(data: List[Dict]) -> List[Document]:
    docs = list() 
    for c_data in data: 
        doc = Document(
            page_content = c_data['page_content'],
            meta_data = {
                'id': c_data['id'], 'url': c_data['url'],
                'date': c_data['date'], 'title': c_data['title']
            }
        )
        docs.append(doc)
    
    return docs 

def build_retrieval_metadata(raw_data):
    return {
        "id": raw_data["id"],
        "url": raw_data["link"], "date": raw_data["pubDate"],
        "title": raw_data["title"]
    }

def build_retrieval_documents(page_content, meta_data) -> Document:
    doc = Document(
        page_content = page_content,
        meta_data = meta_data
    )

    return doc 
