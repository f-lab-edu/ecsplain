import json
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


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