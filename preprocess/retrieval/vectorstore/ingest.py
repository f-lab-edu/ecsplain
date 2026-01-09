
from preprocess.retrieval.config import get_config
from preprocess.retrieval.vectorstore.pipeline import IngestPipeline


def main(config):
    pipeline = IngestPipeline(config)

    pipeline.run()

if __name__ == '__main__': 
    ingest_config = get_config('ingest') 
    main(ingest_config)



    
        




