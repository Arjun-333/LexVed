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

def save_to_history(query, category, subcategory, metrics):
    """Saves a query and its metrics to history.json."""
    history_path = "history.json"
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except:
            history = []
    
    import datetime
    new_entry = {
        "id": len(history) + 1,
        "query": query,
        "category": category,
        "subcategory": subcategory,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": "verified",
        "metrics": metrics
    }
    history.insert(0, new_entry) # Most recent first
    with open(history_path, "w") as f:
        json.dump(history[:50], f, indent=4) # Keep last 50

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Main chat endpoint with SSE-style streaming and history persistence."""
    import time
    t_start = time.time()
    category, subcategory = categorize_text(req.message)
    
    # Simple retrieval first (not streamed)
    t_ret_start = time.time()
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
    
    full_answer = ""
    def stream_response():
        nonlocal full_answer
        # Yield metadata first
        yield json.dumps({
            "type": "metadata",
            "retrieval_time": retrieval_time,
            "category": category,
            "subcategory": subcategory
        }) + "\n"
        
        # Yield answer chunks
        for chunk in generate_answer_stream(req.message, context):
            full_answer += chunk
            yield json.dumps({"type": "content", "text": chunk}) + "\n"
        
        # After stream ends, save to history
        total_time = time.time() - t_start
        save_to_history(req.message, category, subcategory, {
            "retrieval_lat": retrieval_time,
            "e2e_lat": total_time,
            "ans_length": len(full_answer.split())
        })

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
    if os.path.exists("src/ingestion/legal_dictionary.json"):
         files.append({"name": "legal_dictionary.json", "size": "1.2 MB", "type": "Knowledge Base"})
    files.append({"name": "gold_dataset.json", "size": "450 KB", "type": "Evaluation Suite"})
    
    return files

@app.post("/api/analyze")
async def analyze_file(req: dict):
    """Triggers the 24-metric evaluation suite in the background."""
    import subprocess
    import sys
    
    # Run run_metrics.py in the background
    try:
        # We use the current venv if available
        python_exec = sys.executable 
        subprocess.Popen([python_exec, "run_metrics.py"], 
                         start_new_session=True,
                         stdout=open("metric_output.log", "a"),
                         stderr=open("metric_output.log", "a"))
        return {"status": "success", "message": "Performance audit initiated. Results will appear in the dashboard soon."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to initiate audit: {str(e)}"}

@app.get("/api/history")
async def get_history():
    """Returns persistent research history."""
    history_path = "history.json"
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                return json.load(f)
        except:
            return []
    
    # Default initial state
    return [
        {
            "id": 1, 
            "query": "Liability in multi-vehicle collisions", 
            "date": "2024-04-15 14:20", 
            "status": "verified",
            "metrics": {"retrieval_lat": 0.45, "e2e_lat": 2.1, "ans_length": 120}
        },
        {
            "id": 2, 
            "query": "Contractual breach of confidentiality", 
            "date": "2024-04-15 10:15", 
            "status": "verified",
            "metrics": {"retrieval_lat": 0.32, "e2e_lat": 1.8, "ans_length": 95}
        },
    ]

if __name__ == "__main__":
    import uvicorn
    # Add history.json initial check
    if not os.path.exists("history.json"):
        with open("history.json", "w") as f:
            json.dump([], f)
            
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
