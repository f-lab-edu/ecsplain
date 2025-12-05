from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from core.expl.config import config
from core.expl.utils import _ensure_chain, get_answer

app = FastAPI(title="RAG API", version="0.1.0")

# CORS (프론트 도메인으로 제한 권장)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영에선 구체 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_services():
    _ensure_chain(config)


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
        answer = get_answer(q)
        answer, sources = get_answer(q)
        return {"answer": answer, "sources": sources}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
