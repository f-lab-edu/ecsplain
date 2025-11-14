import os
from fastapi import FastAPI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

app = FastAPI()

# --- 서비스 초기화(스타트업 지연 로딩 권장)
_llm = None
_retriever = None

def _format_context(docs):
    # Document 리스트를 하나의 문자열로 합치기 (stuff 방식)
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        parts.append(f"- [{src}] {d.page_content}")
    return "\n".join(parts)

@app.on_event("startup")
def init_services():
    global _llm, _retriever
    if _llm is None:
        _llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"), temperature=0)
        emb = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL","text-embedding-3-small"))
        vs = Chroma(persist_directory=os.getenv("CHROMA_DIR","./chroma_db"), embedding_function=emb)
        _retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

SYSTEM_PROMPT = (
    "너는 신뢰성 높은 도우미야. 주어진 문서 컨텍스트만 사용해 사실에 근거해 답해. "
    "불확실하면 모른다고 말하고, 간단한 출처도 덧붙여."
)
_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "질문: {input}\n\n컨텍스트:\n{context}")
])

# stuff 문서결합 체인
_stuff_chain = (
    {
        "input": RunnablePassthrough(),
        "context": RunnableLambda(lambda x: _format_context(x["context"]))
    }
    | _prompt
    | (lambda p: _llm.invoke(p))  # LLM 호출
    | StrOutputParser()
)

# RAG 파이프라인 (create_retrieval_chain 대체)
def rag_answer(question: str) -> str:
    docs = _retriever.invoke(question)
    return _stuff_chain.invoke({"input": question, "context": docs})

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

from fastapi import Body
@app.post("/query")
def query(payload: dict = Body(...)):
    q = payload.get("question") or payload.get("q") or ""
    if not q.strip():
        return {"error": "question 필드를 입력하세요."}
    ans = rag_answer(q)
    return {"answer": ans}