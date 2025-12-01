import argparse 
import json 
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
from pathlib import Path 
from typing import List, Dict

CHUNK_SIZE = 500 
OVERLAP_SIZE = 100 

def get_config():
    argparser = argparse.ArgumentParser() 
    argparser.add_argument(
        '--emb_model', type=str, default=None
    )
    argparser.add_argument(
        '--data_dir', type=str, default=None
    )
    argparser.add_argument(
        '--chroma_dir', type=str, default='../../storage/vectorstore/chroma_db'
    )
    argparser.add_argument(
        '--openai_api_key', type=str, default=None
    )
    argparser.add_argument(
        '--retrieval_pool_type', type=str, default='default'
    )
    args = argparser.parse_args() 
    if args.openai_api_key is None: 
        args.openai_api_key = os.environ.get('OPENAI_API_KEY', None)

    return args


def load_data(data_dir: Path) -> List[Dict]:
    data: list[Dict] = list() 
    for path in data_dir.rglob('*pool*'):
        if any([
            not path.is_file(),
            'pool' not in path.name
        ]): 
            continue 

        if path.suffix == '.jsonl':
            lines = path.read_text().split('\n')
            for line in lines: 
                js = json.loads(line)
                data.append(js)
        else: 
            raise ValueError(f'unsupported file extension: {path.suffix}')
    
    return data

def split_text(data: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE
    ) 
    split_data = list() 
    for c_data in data:
        text = '\n'.join([c_data['title'], c_data['content']['content']])
        for frag_id, t_frag in enumerate(splitter.split_text(text)):
            f_data = {
                'id': f'{c_data["id"]}.{frag_id}',
                'url': c_data['link'],
                'date': c_data['pubDate'],
                'title': c_data['title'],
                'page_content': t_frag,
            }
            split_data.append(f_data)

    
    return split_data

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

def build_vectorstore(config, docs: List[Document]):
    chroma_dir = Path(config.chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    target_dir = chroma_dir / Path(config.retrieval_pool_type)
    
    embeddings = OpenAIEmbeddings(
        api_key=config.openai_api_key,
        model=config.emb_model
    )

    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory = str(target_dir)
    )


if __name__ == '__main__': 
    config = get_config() 

    # DataLoad
    data_dir = Path(config.data_dir)
    data = load_data(data_dir)

    # Text Split 
    text = split_text(data)
    docs = create_documents(text)


    # Embedding / VectorStore 
    build_vectorstore(config, docs)

    
        




