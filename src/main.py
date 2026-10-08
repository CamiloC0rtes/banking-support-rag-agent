import datetime
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import src.agent as agent_module
from src.agent import blossom_app, call_mcp_holidays
from src.database import run_ingestion

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("blossom_main")

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    history: list[dict] = []
    topic: str | None = None
    user_date: str | None = Field(default_factory=lambda: datetime.date.today().strftime("%Y-%m-%d"))
    temperature: float = 0.2
    top_p: float = 0.9

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    topic: str | None
    history: list[dict]
    timing_ms: float
    grounded: bool = False
    sources: list[dict] = []
    node_latency_ms: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """System warm-up and state hydration sequence."""
    logger.info("Initializing Blossom AI Engine...")
    try:
        # 1. Database persistence check
        run_ingestion(force_rebuild=False)
        
        # 2. Global MCP cache hydration (source: federal holiday API)
        agent_module._CACHED_HOLIDAYS = await call_mcp_holidays()
        
        # 3. Agent warm-up to mitigate cold-start latency
        await blossom_app.ainvoke({
            "message": "warmup", 
            "user_date": datetime.date.today().strftime("%Y-%m-%d"), 
            "history": [],
            "temperature": 0.1
        })
        logger.info("Warm-up complete. System is online.")
    except Exception as e:
        logger.error(f"Warm-up sequence failed: {e}")
    yield

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    """Middleware for real-time SLA tracking and observability."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    
    # Custom headers for monitoring
    response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))
    response.headers["X-SLA-Status"] = "MET" if process_time < 5000 else "BREACHED"
    return response

@app.get("/")
async def get_index(): 
    """Serves the main web interface."""
    static_path = os.path.join(os.getcwd(), 'src', 'static', 'index.html')
    return FileResponse(static_path)

@app.get("/health")
async def health(): 
    """Liveness probe for infrastructure health monitoring."""
    return {
        "status": "healthy", 
        "timestamp": datetime.datetime.now().isoformat(),
        "mcp_ready": agent_module._CACHED_HOLIDAYS is not None,
        "current_date": datetime.date.today().strftime("%Y-%m-%d") # source: federal holiday API
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    start_time = time.perf_counter()
    result = await blossom_app.ainvoke(req.model_dump())
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    return {
        "answer": result["answer"],
        "session_id": req.session_id or f"sess_{int(time.time())}",
        "topic": result.get("topic"),
        "history": result.get("history", []),
        "timing_ms": round(duration_ms, 2),
        "grounded": result.get("grounded", False),
        "sources": result.get("sources", []),
        "node_latency_ms": result.get("node_latency_ms", {}),
    }


STREAMING_NODES = {"generate", "greet", "farewell"}


@app.get("/chat/stream")
async def chat_stream(message: str, user_date: str | None = None):
    """SSE endpoint. Runs the same graph as /chat (routing, RAG, grounding gate)
    and streams LLM tokens as they are produced."""
    state = {
        "message": message,
        "history": [],
        "user_date": user_date or datetime.date.today().strftime("%Y-%m-%d"),
    }

    async def event_generator():
        final, streamed = {}, False
        async for mode, chunk in blossom_app.astream(state, stream_mode=["messages", "values"]):
            if mode == "messages":
                msg, meta = chunk
                if meta.get("langgraph_node") in STREAMING_NODES and msg.content:
                    streamed = True
                    yield f"data: {json.dumps({'token': msg.content})}\n\n"
            else:
                final = chunk
        if not streamed and final.get("answer"):
            # fallback is a fixed template, not an LLM call
            yield f"data: {json.dumps({'token': final['answer']})}\n\n"
        yield f"data: {json.dumps({'sources': final.get('sources', []), 'grounded': final.get('grounded', False)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")