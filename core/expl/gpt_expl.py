import openai
import os 
import json
import argparse
import tqdm
import time
import random

from utils import ensure_chain, get_answer

def get_config():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_path', type=str, default='prompts/expl/con_detailed.txt')
    argparser.add_argument('--save_fp', type=str, default='results/gpt4_con_detailed_openai.json')
    argparser.add_argument('--expl_fp', type=str, default='data/expl.json')
    
    argparser.add_argument('--api_key', type=str, default=None)
    argparser.add_argument('--embedding_model', type=str, default='text-embedding-3-small')
    argparser.add_argument('--openai_model', type=str, default='gpt-5')
    argparser.add_argument('--retrieval_k', type=int, default=5)
    argparser.add_argument('--reasoning_effort', type=str, default='high')
    argparser.add_argument('--chroma_dir', type=str, default='.')
    argparser.add_argument('--data_dir', type=str, default='.')

    argparser.add_argument('--sample_num', type=int, default=0)
    

    return argparser.parse_args()

def main(config):
    if config.api_key is None: 
        config.api_key = os.environ.get('OPENAI_API_KEY')
    ensure_chain(config)

    data = read_data(config)
    for instance in tqdm.tqdm(data):
        while True:
            try:
                explanation = get_answer(instance['source'])
                time.sleep(0.5)
                instance['explanation'] = explanation
                break
            except Exception as e: 
                print(e)
                if ("limit" not in str(e)): break 
                time.sleep(2)

    write_data(config, data)


def read_data(config):
    with open(config.expl_fp, 'r') as read_fp:  
        data = list()
        if 'jsonl' in config.expl_fp:
            for line in read_fp.readlines(): 
                js = json.loads(line)
                data.append(js) 
            
        else:
            data = json.load(read_fp)
    
    if config.sample_num > 0: 
        data = random.sample(data, k=config.sample_num)
    
    return data

def write_data(config, data): 
    base_dir = '/'.join(config.save_fp.split('/')[:-1])
    os.makedirs(base_dir, exist_ok=True)

    with open(config.save_fp, 'w', encoding='utf-8') as write_fp: 
        json.dump(data, write_fp, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    config = get_config()
    main(config)


