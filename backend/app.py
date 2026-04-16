from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.utils.config_manager import set_active_model, get_active_model_name, get_active_model_params
import subprocess

from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer_stream, generate_answer
from src.ingestion.pdf_processor import categorize_text

app = FastAPI(title="LexVed API", version="2.0")

# CORS - allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-bound tasks (retrieval, generation)
executor = ThreadPoolExecutor(max_workers=4)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    retrieval_time: float
    generation_time: float
    category: str
    subcategory: str
    provider: str

def _process_chat(question: str) -> dict:
    """Synchronous chat processing (runs in thread pool)."""
    category, subcategory = categorize_text(question)
    print(f"[LexVed] Query: {question}")
    print(f"[LexVed] Category: {category}, Subcategory: {subcategory}")

    # Retrieve context from Qdrant with filters
    res, retrieval_time = retrieve(
        question,
        top_k=5,
        category=category if category != "Uncategorized" else None,
        subcategory=subcategory if subcategory != "General" else None
    )

    # Build context string
    context = ""
    for m in res:
        source = os.path.basename(m.payload.get("source", "Unknown"))
        page = m.payload.get("page", "?")
        context += f"\n[Source: {source}, Page: {page}]\n{m.payload['text']}\n"

    # Fallback: unfiltered search if no results
    if not context.strip():
        res, _ = retrieve(question, top_k=5)
        for m in res:
            source = os.path.basename(m.payload.get("source", "Unknown"))
            page = m.payload.get("page", "?")
            context += f"\n[Source: {source}, Page: {page}]\n{m.payload['text']}\n"

    # Generate with Llama 3
    answer, generation_time, _ = generate_answer(question, context)

    return {
        "response": answer,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "category": category,
        "subcategory": subcategory,
        "provider": "Llama 3 (Local)"
    }

from fastapi.responses import StreamingResponse
import json

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Main chat endpoint with SSE-style streaming."""
    category, subcategory = categorize_text(req.message)
    
    # Simple retrieval first (not streamed)
    res, retrieval_time = retrieve(
        req.message,
        top_k=5,
        category=category if category != "Uncategorized" else None,
        subcategory=subcategory if subcategory != "General" else None
    )

    context = ""
    for m in res:
        source = os.path.basename(m.payload.get("source", "Unknown"))
        page = m.payload.get("page", "?")
        context += f"\n[Source: {source}, Page: {page}]\n{m.payload['text']}\n"

    from src.generation.generator import generate_answer_stream
    
    def stream_response():
        # Yield metadata first
        yield json.dumps({
            "type": "metadata",
            "retrieval_time": retrieval_time,
            "category": category,
            "subcategory": subcategory
        }) + "\n"
        
        # Yield answer chunks
        for chunk in generate_answer_stream(req.message, context):
            yield json.dumps({"type": "content", "text": chunk}) + "\n"

    return StreamingResponse(stream_response(), media_type="application/x-ndjson")

@app.get("/api/metrics")
async def get_metrics():
    """Returns the latest performance audit results."""
    metric_path = "evaluation_results.json"
    if os.path.exists(metric_path):
        with open(metric_path, "r") as f:
            return json.load(f)
    return {"status": "error", "message": "No audit report found. Please run run_metrics.py first."}

class ModelSettings(BaseModel):
    model: str

@app.get("/api/settings/embedding_model")
async def get_model_setting():
    return {"model": get_active_model_name()}

@app.post("/api/settings/embedding_model")
async def set_model_setting(settings: ModelSettings):
    success = set_active_model(settings.model)
    if success:
        return {"status": "success", "model": settings.model}
    return {"status": "error", "message": "Invalid model name"}

@app.post("/api/workflow/evaluate")
async def trigger_evaluation():
    """Triggers a full re-ingestion and evaluation cycle."""
    # We run this in the background to avoid blocking the API
    def run_eval():
        print("[LexVed] Starting Evaluation Workflow...")
        # 1. Update status to processing immediately
        try:
            with open("evaluation_results.json", "w") as f:
                json.dump({"status": "processing", "progress": "Initializing Intelligence Node..."}, f)
        except:
            pass

        # 2. Re-initialize Vector DB
        try:
            from src.utils.qdrant_client import init_collection as init_qdrant
            init_qdrant()
            print("[LexVed] Qdrant initialized.")
        except:
            pass
        
        # 3. Run Metrics
        try:
            print("[LexVed] Running Benchmark...")
            # Use the same venv as the app
            subprocess.Popen(["./venv/bin/python3", "run_metrics.py"])
        except Exception as e:
            print(f"[LexVed] Benchmark error: {e}")
            with open("evaluation_results.json", "w") as f:
                json.dump({"status": "error", "message": str(e)}, f)

    executor.submit(run_eval)
    return {"status": "processing", "message": "Evaluation started in background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
