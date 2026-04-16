from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import subprocess

from src.utils.config_manager import set_active_model, get_active_model_name, get_active_model_params, get_active_db_name, set_active_db
from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer_stream, generate_answer
from src.ingestion.pdf_processor import categorize_text

app = FastAPI(title="LexVed API", version="2.0")

# CORS - allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

class ChatRequest(BaseModel):
    message: str

class ModelSettings(BaseModel):
    model: str

class DBSettings(BaseModel):
    db: str

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
    history.insert(0, new_entry)
    with open(history_path, "w") as f:
        json.dump(history[:50], f, indent=4)

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Main chat endpoint with SSE-style streaming."""
    import time
    t_start = time.time()
    category, subcategory = categorize_text(req.message)
    
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

    full_answer = ""
    def stream_response():
        nonlocal full_answer
        yield json.dumps({
            "type": "metadata",
            "retrieval_time": retrieval_time,
            "category": category,
            "subcategory": subcategory,
            "vector_db": get_active_db_name()
        }) + "\n"
        
        for chunk in generate_answer_stream(req.message, context):
            full_answer += chunk
            yield json.dumps({"type": "content", "text": chunk}) + "\n"
        
        total_time = time.time() - t_start
        save_to_history(req.message, category, subcategory, {
            "retrieval_lat": retrieval_time,
            "e2e_lat": total_time,
            "ans_length": len(full_answer.split())
        })

    return StreamingResponse(stream_response(), media_type="application/x-ndjson")

@app.get("/api/metrics")
async def get_metrics():
    import psutil
    metric_path = "evaluation_results.json"
    if os.path.exists(metric_path):
        try:
            with open(metric_path, "r") as f:
                report = json.load(f)
            
            # Check if process is stuck
            if report.get("status") == "processing":
                pid = report.get("pid")
                if pid:
                    try:
                        p = psutil.Process(pid)
                        if not p.is_running():
                             report["status"] = "error"
                             report["message"] = "Process died unexpectedly."
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        report["status"] = "error"
                        report["message"] = "Process no longer active."
                else:
                    # No PID, check if it's too old
                    import time
                    # We'll allow 5 mins baseline
                    pass 
            return report
        except Exception as e:
            return {"status": "error", "message": f"Read error: {e}"}
    return {"status": "error", "message": "No audit report found."}

@app.get("/api/history")
async def get_history():
    history_path = "history.json"
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                return json.load(f)
        except:
            return []
    return []

@app.get("/api/settings/embedding_model")
async def get_model_setting():
    return {"model": get_active_model_name()}

@app.post("/api/settings/embedding_model")
async def set_model_setting(settings: ModelSettings):
    if set_active_model(settings.model):
        return {"status": "success", "model": settings.model}
    return {"status": "error", "message": "Invalid model"}

@app.get("/api/settings/vector_db")
async def get_db_setting():
    return {"db": get_active_db_name()}

@app.post("/api/settings/vector_db")
async def set_db_setting(settings: DBSettings):
    if set_active_db(settings.db):
        return {"status": "success", "db": settings.db}
    return {"status": "error", "message": "Invalid database"}

@app.post("/api/workflow/evaluate")
async def trigger_evaluation():
    def run_eval():
        print(f"[LexVed] Starting Evaluation Workflow on {get_active_db_name()}...")
        
        # Initialize DB
        active_db = get_active_db_name()
        try:
            if active_db == "qdrant":
                from src.utils.qdrant_provider import init_collection
                init_collection()
            else:
                from src.utils.pinecone_client import create_index
                create_index()
        except Exception as e:
            print(f"[LexVed] DB Init error: {e}")
            with open("evaluation_results.json", "w") as f:
                json.dump({"status": "error", "message": f"DB Init failed: {e}"}, f)
            return

        # Run Metrics
        try:
            proc = subprocess.Popen(["./venv/bin/python3", "run_metrics.py"])
            with open("evaluation_results.json", "w") as f:
                json.dump({
                    "status": "processing", 
                    "progress": "Initializing Intelligence Node...",
                    "pid": proc.pid
                }, f)
        except Exception as e:
            with open("evaluation_results.json", "w") as f:
                json.dump({"status": "error", "message": str(e)}, f)

    executor.submit(run_eval)
    # Give it a tiny bit of time to create the file with PID
    return {"status": "processing", "message": "Evaluation initiated"}

if __name__ == "__main__":
    import uvicorn
    if not os.path.exists("history.json"):
        with open("history.json", "w") as f: json.dump([], f)
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
