from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

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

@app.get("/api/files")
async def list_files():
    """Lists legal dictionary and corpus files."""
    files = []
    # Mock some data if folders don't exist, but try to list real directories
    data_dir = "data"
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith((".pdf", ".docx", ".txt")):
                files.append({"name": f, "size": f"{os.path.getsize(os.path.join(data_dir, f))/1024:.1f} KB", "type": "Legal Document"})
    
    # Add system files for context
    files.append({"name": "legal_dictionary.json", "size": "1.2 MB", "type": "Knowledge Base"})
    files.append({"name": "gold_dataset.json", "size": "450 KB", "type": "Evaluation Suite"})
    
    return files

@app.get("/api/history")
async def get_history():
    """Returns mock research history based on evaluation logs."""
    history = [
        {"id": 1, "query": "Liability in multi-vehicle collisions", "date": "2 hours ago", "status": "verified"},
        {"id": 2, "query": "Contractual breach of confidentiality", "date": "Yesterday", "status": "verified"},
        {"id": 3, "query": "Intellectual property infringement in software", "date": "Oct 12", "status": "flagged"},
        {"id": 4, "query": "Medical malpractice statute of limitations", "date": "Oct 10", "status": "verified"},
    ]
    return history

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
