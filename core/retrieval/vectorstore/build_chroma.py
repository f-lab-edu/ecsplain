import yaml 
from pathlib import Path 
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings 

BASE_DIR = Path(__file__).resolve().parents[3]


def load_pool_infos(info_path):
    with open(info_path, 'r') as info_fp: 
        infos = yaml.safe_load(info_fp)
    
    return infos

def load_retriever(pool_infos, config):
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
