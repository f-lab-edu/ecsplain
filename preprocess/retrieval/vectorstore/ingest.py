import argparse 
import json 
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
from pathlib import Path 
from typing import List, Dict

from preprocess.retrieval.storage import LocalStorage, S3Storage
from preprocess.retrieval.config import get_config
from preprocess.retrieval.vectorstore.pipeline import LoadRawDataStep, SplitStep, BuildVectorStoreStep
from preprocess.retrieval.vectorstore.pipeline import IngestPipeline


def main(config):
    import pdb;pdb.set_trace()
    pipeline = IngestPipeline(config)

    ctx = pipeline.run()

if __name__ == '__main__': 
    ingest_config = get_config('ingest') 
    main(ingest_config)



    
        




