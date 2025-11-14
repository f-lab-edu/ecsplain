import os
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

app = FastAPI(title="RAG API", version="0.1.0")

# CORS (프론트 도메인으로 제한 권장)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영에선 구체 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 지연 초기화
_LLM = None
_RETRIEVER = None

def _format_context(docs):
    if not docs:
        return "(관련 문서 없음)"
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        lines.append(f"- [{src}] {d.page_content}")
    return "\n".join(lines)

def _ensure_services():
    global _LLM, _RETRIEVER
    if _LLM is not None and _RETRIEVER is not None:
        return
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(500, "OPENAI_API_KEY 미설정")

    _LLM = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    emb = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    vs = Chroma(
        persist_directory=os.getenv("CHROMA_DIR", "./chroma_db"),
        embedding_function=emb
    )
    # LangChain 0.2+ retriever는 Runnable: invoke/ainvoke 사용
    _RETRIEVER = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

SYSTEM_PROMPT = (
    "너는 신뢰성 높은 도우미야. 주어진 문서 컨텍스트만 사용해 사실에 근거해 답해. "
    "불확실하면 모른다고 말하고, 간단한 출처도 덧붙여."
)
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "질문: {input}\n\n컨텍스트:\n{context}")
])

STUFF_CHAIN = (
    {
        "input": RunnablePassthrough(),
        "context": RunnableLambda(lambda x: _format_context(x["context"]))
    }
    | PROMPT
    | (lambda p: _LLM.invoke(p))
    | StrOutputParser()
)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>RAG API</h1>
    <ul>
      <li><a href="/docs">/docs</a></li>
      <li><a href="/healthz">/healthz</a></li>
    </ul>
    """

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/query")
def query(payload: dict = Body(...)):
    try:
        q = (payload.get("question") or payload.get("q") or "").strip()
        if not q:
            raise HTTPException(400, "`question` 필드가 필요합니다.")
        _ensure_services()
        docs = _RETRIEVER.invoke(q)
        answer = STUFF_CHAIN.invoke({"input": q, "context": docs})
        sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs]
        return {"answer": answer, "sources": sources}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})