from fastapi import FastAPI, UploadFile, File, Request
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
import time as time_module
from typing import Optional, List

from src.utils.config_manager import (
    set_active_model, get_active_model_name, get_active_model_params,
    get_active_db_name, set_active_db, get_active_generation_model,
    set_active_generation_model, load_config
)
from src.retrieval.retriever import retrieve, invalidate_bm25
from src.generation.generator import generate_answer_stream, generate_answer
from src.ingestion.pdf_processor import categorize_text

load_dotenv()

app = FastAPI(title="LexVed API", version="3.0")

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

# Track server uptime
_server_start_time = time_module.time()

# ─── Request Models ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = None  # Multi-turn conversation memory

class ModelSettings(BaseModel):
    model: str

class DBSettings(BaseModel):
    db: str

# ─── Utility ──────────────────────────────────────────────────────

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

# ─── Chat Endpoint (Multi-Turn) ──────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Main chat endpoint with SSE-style streaming and multi-turn memory."""
    t_start = time_module.time()
    category, subcategory = categorize_text(req.message)
    
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
        
        for chunk in generate_answer_stream(req.message, context, history=req.history):
            full_answer += chunk
            yield json.dumps({"type": "content", "text": chunk}) + "\n"
        
        total_time = time_module.time() - t_start
        save_to_history(req.message, category, subcategory, {
            "retrieval_lat": retrieval_time,
            "e2e_lat": total_time,
            "ans_length": len(full_answer.split())
        })

    return StreamingResponse(stream_response(), media_type="application/x-ndjson")

# ─── Health Check ─────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Returns system health: Ollama, Qdrant, active model, uptime."""
    import requests
    health = {
        "status": "operational",
        "uptime_seconds": round(time_module.time() - _server_start_time, 1),
        "active_generation_model": get_active_generation_model(),
        "active_embedding_model": get_active_model_name(),
        "active_vector_db": get_active_db_name(),
        "ollama": "unknown",
        "vector_db": "unknown"
    }
    
    # Check Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            health["ollama"] = "connected"
            health["ollama_models"] = models
        else:
            health["ollama"] = "error"
    except:
        health["ollama"] = "offline"

    # Check vector DB
    active_db = get_active_db_name()
    if active_db == "qdrant":
        try:
            from qdrant_client import QdrantClient
            c = QdrantClient(host="localhost", port=6333, timeout=3)
            cols = [col.name for col in c.get_collections().collections]
            health["vector_db"] = "connected"
            health["collections"] = cols
        except:
            health["vector_db"] = "offline"
    else:
        try:
            from src.utils.pinecone_client import index
            stats = index.describe_index_stats()
            health["vector_db"] = "connected"
            health["vector_count"] = stats.get('total_vector_count', 0)
        except:
            health["vector_db"] = "offline"

    return health

# ─── Metrics & History ────────────────────────────────────────────

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

@app.delete("/api/history")
async def clear_history():
    """Clears the query history log."""
    history_path = "history.json"
    with open(history_path, "w") as f:
        json.dump([], f)
    return {"status": "success", "message": "History cleared."}

# ─── File Management & Ingestion ──────────────────────────────────

@app.get("/api/files")
async def list_files():
    """Lists all PDF files in the data directory."""
    pdf_dir = "data/PDF"
    files = []
    if os.path.exists(pdf_dir):
        for fname in os.listdir(pdf_dir):
            if fname.lower().endswith(".pdf"):
                fpath = os.path.join(pdf_dir, fname)
                size_bytes = os.path.getsize(fpath)
                if size_bytes > 1024 * 1024:
                    size_str = f"{size_bytes / (1024*1024):.1f} MB"
                else:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                files.append({
                    "name": fname,
                    "size": size_str,
                    "type": "Legal Document"
                })
    
    # Also list evaluation data and knowledge base files
    if os.path.exists("evaluation_data.json"):
        files.append({"name": "evaluation_data.json", "size": f"{os.path.getsize('evaluation_data.json') / 1024:.1f} KB", "type": "Evaluation Suite"})
    if os.path.exists("evaluation_results.json"):
        files.append({"name": "evaluation_results.json", "size": f"{os.path.getsize('evaluation_results.json') / 1024:.1f} KB", "type": "Knowledge Base"})
    
    return files

@app.post("/api/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF into the vector database.
    Returns progress via streaming response.
    """
    pdf_dir = "data/PDF"
    os.makedirs(pdf_dir, exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join(pdf_dir, file.filename)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    def ingest_stream():
        try:
            from src.ingestion.pdf_processor import extract_chunks, process_chunks_batch
            from src.ingestion.embedder import get_embeddings
            
            yield json.dumps({"status": "processing", "step": "Extracting text from PDF..."}) + "\n"
            chunks = extract_chunks(file_path)
            yield json.dumps({"status": "processing", "step": f"Extracted {len(chunks)} chunks. Processing NER..."}) + "\n"
            
            chunks = process_chunks_batch(chunks)
            yield json.dumps({"status": "processing", "step": f"Cleaned {len(chunks)} chunks. Generating embeddings..."}) + "\n"
            
            # Categorize and embed in batches
            texts = [c["text"] for c in chunks]
            embeddings = get_embeddings(texts)
            yield json.dumps({"status": "processing", "step": f"Generated {len(embeddings)} embeddings. Upserting to {get_active_db_name()}..."}) + "\n"
            
            active_db = get_active_db_name()
            if active_db == "qdrant":
                from qdrant_client import QdrantClient
                from qdrant_client.models import PointStruct
                from src.utils.qdrant_provider import COLLECTION_NAME
                client = QdrantClient(host="localhost", port=6333)
                
                points = []
                import uuid
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    cat, sub = categorize_text(chunk["text"])
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb.tolist(),
                        payload={
                            "text": chunk["text"],
                            "source": chunk["source"],
                            "page": chunk["page"],
                            "category": cat,
                            "subcategory": sub
                        }
                    ))
                
                # Batch upsert (100 at a time)
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch = points[i:i+batch_size]
                    client.upsert(collection_name=COLLECTION_NAME, points=batch)
                    yield json.dumps({"status": "processing", "step": f"Upserted {min(i+batch_size, len(points))}/{len(points)} vectors..."}) + "\n"
            else:
                from src.utils.pinecone_client import index
                import uuid
                vectors = []
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    cat, sub = categorize_text(chunk["text"])
                    vectors.append({
                        "id": str(uuid.uuid4()),
                        "values": emb.tolist(),
                        "metadata": {
                            "text": chunk["text"],
                            "source": chunk["source"],
                            "page": chunk["page"],
                            "category": cat,
                            "subcategory": sub
                        }
                    })
                
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i+batch_size]
                    index.upsert(vectors=batch)
                    yield json.dumps({"status": "processing", "step": f"Upserted {min(i+batch_size, len(vectors))}/{len(vectors)} vectors..."}) + "\n"
            
            # Invalidate BM25 so it rebuilds with new data
            invalidate_bm25()
            
            yield json.dumps({
                "status": "complete",
                "step": f"Successfully ingested {file.filename}: {len(chunks)} chunks indexed.",
                "chunks": len(chunks)
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({"status": "error", "step": f"Ingestion failed: {str(e)}"}) + "\n"

    return StreamingResponse(ingest_stream(), media_type="application/x-ndjson")

# ─── Settings ─────────────────────────────────────────────────────

@app.get("/api/settings/embedding_model")
async def get_model_setting():
    return {"model": get_active_model_name()}

@app.post("/api/settings/embedding_model")
async def set_model_setting(settings: ModelSettings):
    if set_active_model(settings.model):
        invalidate_bm25()  # Invalidate caches on model switch
        return {"status": "success", "model": settings.model}
    return {"status": "error", "message": "Invalid model"}

@app.get("/api/settings/vector_db")
async def get_db_setting():
    return {"db": get_active_db_name()}

@app.post("/api/settings/vector_db")
async def set_db_setting(settings: DBSettings):
    if set_active_db(settings.db):
        invalidate_bm25()  # Invalidate caches on DB switch
        return {"status": "success", "db": settings.db}
    return {"status": "error", "message": "Invalid database"}

@app.get("/api/settings/generation_model")
async def get_gen_model_setting():
    return {"model": get_active_generation_model()}

@app.post("/api/settings/generation_model")
async def set_gen_model_setting(settings: ModelSettings):
    if set_active_generation_model(settings.model):
        return {"status": "success", "model": settings.model}
    return {"status": "error", "message": "Invalid generation model"}

@app.get("/api/settings/config")
async def get_full_config():
    """Returns the full config for frontend consumption."""
    config = load_config()
    return {
        "embedding_models": list(config.get("models", {}).keys()),
        "generation_models": config.get("generation_models", []),
        "providers": config.get("providers", [])
    }

# ─── Evaluation Workflows ────────────────────────────────────────

@app.post("/api/workflow/evaluate")
async def trigger_evaluation():
    def run_eval():
        print(f"[LexVed] Starting Evaluation Workflow on {get_active_db_name()}...")
        # Run Metrics

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
    return {"status": "processing", "message": "Evaluation initiated"}

@app.post("/api/workflow/comparative")
async def trigger_comparative(request: Request):
    """Triggers comparative benchmarking across all embedding models."""
    body = await request.json()
    resume = body.get("resume", False)
    
    def run_comparative_thread():
        try:
            cmd = ["./venv/bin/python3", "run_comparative.py"]
            if resume:
                cmd.append("--resume")
                
            proc = subprocess.Popen(cmd)
            
            if not resume:
                if os.path.exists("intermediate_results.json"):
                    try: os.remove("intermediate_results.json")
                    except: pass

                with open("comparative_results.json", "w") as f:
                    json.dump({
                        "status": "processing",
                        "progress": "Starting comparative benchmark...",
                        "completed_models": [],
                        "current_model": "",
                        "pid": proc.pid
                    }, f)
            else:
                if os.path.exists("comparative_results.json"):
                    try:
                        with open("comparative_results.json", "r") as f:
                            old_data = json.load(f)
                        old_data["status"] = "processing"
                        old_data["pid"] = proc.pid
                        with open("comparative_results.json", "w") as f:
                            json.dump(old_data, f, indent=2)
                    except: pass
        except Exception as e:
            with open("comparative_results.json", "w") as f:
                json.dump({"status": "error", "message": str(e)}, f)

    executor.submit(run_comparative_thread)
    return {"status": "processing", "message": "Comparative benchmark initiated"}

@app.get("/api/comparative")
async def get_comparative():
    """Returns comparative benchmarking results."""
    path = "comparative_results.json"
    if os.path.exists(path):
        try:
            import psutil
            with open(path, "r") as f:
                report = json.load(f)
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
            return report
        except Exception as e:
            return {"status": "error", "message": f"Failed to read comparative results: {e}"}
    return {"status": "error", "message": "No comparative results found. Run a comparative benchmark first."}

# ─── Server Entry ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    if not os.path.exists("history.json"):
        with open("history.json", "w") as f: json.dump([], f)
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
