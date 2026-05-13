"""HTTP API for the Amharic RAG stack (OpenAPI / Swagger at ``/docs``).

Run from project root::

    .venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Env:
    RAG_CHROMA_DB — path to Chroma persist dir (default: ./chroma_amharic)
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import chromadb

from llm_providers import effective_llm_backend, load_dotenv_if_present
from query import default_top_k, run_query

load_dotenv_if_present()

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "chroma_amharic"

app = FastAPI(
    title="Amharic Farmer RAG API",
    version="1.0.0",
    description=(
        "Retrieve + generate over the local Chroma index (PDFs, merged.json Q&A, optional dynamic chunks). "
        "Requires a built index: ``python ingest.py``. LLM: Groq / Gemini / Ollama / OpenAI via env (see ``GET /v1/llm``)."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _chroma_db() -> Path:
    return Path(os.environ.get("RAG_CHROMA_DB", str(DEFAULT_DB))).resolve()


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=12000)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    top_k: int | None = Field(None, ge=2, le=32, description="Overrides default (4 fast / 6 quality)")
    fast: bool = Field(True, description="Fast preset vs quality (larger context / model)")
    show_sources: bool = Field(True, description="Include retrieval source previews")
    retrieve_only: bool = Field(
        False,
        description="Skip LLM; return top chunks only (no Ollama/API needed)",
    )
    conversation: list[ChatMessage] | None = Field(
        None,
        description="Prior user/assistant turns for multi-turn chat",
    )


@app.get("/v1/llm", tags=["Health"])
def v1_llm() -> dict:
    """Resolved backend after ``RAG_LLM_BACKEND`` / API keys (same logic as ``query.run_query``)."""
    return {"llm_backend": effective_llm_backend()}


@app.get("/health", tags=["Health"])
def health() -> dict:
    db = _chroma_db()
    chroma_ok = False
    if db.is_dir():
        try:
            client = chromadb.PersistentClient(path=str(db))
            client.get_collection("amharic_rag")
            chroma_ok = True
        except Exception:
            chroma_ok = False
    return {
        "status": "ok" if chroma_ok else "degraded",
        "chroma_db": str(db),
        "index_present": chroma_ok,
    }


@app.post("/v1/query", tags=["Query"])
def v1_query(req: QueryRequest) -> dict:
    db = _chroma_db()
    if not db.is_dir():
        raise HTTPException(
            status_code=503,
            detail=f"Chroma directory missing: {db}. Run: python ingest.py",
        )
    top_k = req.top_k if req.top_k is not None else default_top_k(req.fast)
    conv = None
    if req.conversation:
        conv = [m.model_dump() for m in req.conversation]
    try:
        return run_query(
            req.question.strip(),
            db,
            top_k,
            req.show_sources,
            retrieve_only=req.retrieve_only,
            fast=req.fast,
            conversation=conv,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/retrieve", tags=["Query"])
def v1_retrieve(req: QueryRequest) -> dict:
    """Same body as ``/v1/query`` but forces ``retrieve_only=true``."""
    req2 = req.model_copy(update={"retrieve_only": True})
    return v1_query(req2)
