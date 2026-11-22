"""
FastAPI Backend Server for Enterprise Developer Chatbot
======================================================
Connects the LangGraph RAG pipeline in main.py to the Web UI.
Serves static frontend assets and provides the /api/chat endpoint.
"""

import os
import sys
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure workspace is on sys.path
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from main import build_pipeline, vs, fast_llm, power_llm, FAST_MODEL, POWER_MODEL, KB_SCORE_THRESHOLD, WEB_SCORE_THRESHOLD

load_dotenv()

app = FastAPI(
    title="Enterprise Developer Chatbot API",
    description="Full-Stack Documentation Engine & Developer Assistant",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline as singleton
print("Initializing LangGraph pipeline in server...")
pipeline = build_pipeline()

@app.on_event("startup")
def on_startup():
    print("Pre-warming vector store connection and embedding model...")
    try:
        vs()
        print("Vector store connection and embedding model pre-warmed.")
    except Exception as e:
        print(f"Warmup warning: {e}")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    source: str
    kb_score: Optional[int] = None
    kb_score_detail: Optional[str] = None
    web_score: Optional[int] = None
    web_score_detail: Optional[str] = None
    duration: float
    timestamp: float

@app.get("/api/status")
def get_status():
    """Returns engine telemetry and active model configurations."""
    return {
        "status": "online",
        "fast_model": FAST_MODEL,
        "power_model": POWER_MODEL,
        "kb_threshold": KB_SCORE_THRESHOLD,
        "web_threshold": WEB_SCORE_THRESHOLD,
        "collection": "fullstack_devdocs",
        "catalog": ["React", "FastAPI", "Python", "JavaScript", "TypeScript", "HTML", "CSS", "REST APIs", "Git"]
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """Processes user queries through the guardrail -> Qdrant -> eval -> Tavily pipeline."""
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    t0 = time.time()
    try:
        result = pipeline.invoke({"question": q})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")
    duration = round(time.time() - t0, 2)

    return ChatResponse(
        question=q,
        answer=result.get("answer", "No answer generated."),
        source=result.get("source", "unknown"),
        kb_score=result.get("kb_score"),
        kb_score_detail=result.get("kb_score_detail"),
        web_score=result.get("web_score"),
        web_score_detail=result.get("web_score_detail"),
        duration=duration,
        timestamp=time.time()
    )

# Static files will be mounted at root
static_dir = os.path.join(WORKSPACE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
