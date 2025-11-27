from pathlib import Path

from fastapi import HTTPException
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parents[2]

_LLM, _RETRIEVER = None, None
_STUFF_CHAIN = None 

def _format_context(docs):
    if not docs: 
        return '(관련 문서 없음)'

    lines = list() 
    for doc in docs: 
        src = doc.metadata.get('source', 'unknown')
        lines.append(f'- [{src}] {doc.page_content}')
    
    return '\n'.join(lines)

def get_variables():
    variables = {
        'input': RunnablePassthrough(),
        'context': RunnableLambda(lambda x: _format_context(x['context']))
    } 

    return variables

def get_prompt_template(prompt_path):
    with open(prompt_path, 'r') as read_fp:
        raw_template = read_fp.read()
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("human", raw_template)
    ])

    return prompt_template

def set_models(config):
    global _RETRIEVER, _LLM
    if (_LLM is not None) and (_RETRIEVER is not None): 
        return
    
    api_key = config.openai_api_key 
    if not api_key: 
        raise HTTPException(500, "OPENAI_API_KEY 미설정")
    
    if 'gpt-5' in config.openai_model:
        _LLM = ChatOpenAI(
            model = config.openai_model, 
            api_key = api_key,
            reasoning = { 'effort': config.reasoning_effort } 
        )
    else: 
        _LLM = ChatOpenAI(
            model = config.openai_model,
            api_key = api_key,
            temperature = config.temperature
        )

    emb_model = OpenAIEmbeddings(
        model=config.embedding_model,
        api_key = api_key
    )

    vector_store = Chroma(
        embedding_function=emb_model,
        persist_directory=str(
           BASE_DIR / Path(config.chroma_dir)
        )
    )
    _RETRIEVER = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": config.retrieval_k}
    )

def construct_chain(config):
    global _STUFF_CHAIN

    variables = get_variables()
    prompt_template = get_prompt_template(config.prompt_path)
    output_parser = StrOutputParser() 

    _STUFF_CHAIN = (
        variables 
        | prompt_template 
        | (lambda p: _LLM.invoke(p))
        | output_parser
    )

def _ensure_chain(config):
    set_models(config)
    construct_chain(config) 

def get_retrieval(query):
    docs = _RETRIEVER.invoke(query)

    return docs


def get_answer(query):
    docs = get_retrieval(query)
    answer = _STUFF_CHAIN.invoke({'input': query, 'context': docs})
    sources =[
        {'source': d.metadata.get('source'), 'page': d.metadata.get('page')} for d in docs 
    ]

    return answer, sources

