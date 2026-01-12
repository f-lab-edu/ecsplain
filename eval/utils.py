import hashlib
import json
import subprocess

from langchain_core.documents import Document


def detect_git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""

def canonical_json(obj) -> str: 
    return json.dumps(
        obj,
        sort_keys=True, 
        separators=(",", ":"),
        ensure_ascii=False
    )

def make_config_hash(config_dict: dict) -> str:
    s = canonical_json(config_dict)

    return hashlib.sha1(s.encode("utf-8")).hexdigest()
    

def bytes_to_jsonl(byte_seq):
    lines = byte_seq.decode('utf-8').strip().split('\n')

    jsonl = list() 
    for line in lines:
        js = json.loads(line)
        jsonl.append(js)
    
    return jsonl


def bytes_to_docs(byte_seq):
    data = byte_seq.decode('utf-8')
    docs = list()
    for line_id, line in enumerate(data.split('\n')):
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

