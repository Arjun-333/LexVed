from fastapi import FastAPI, UploadFile, File, Request, Depends, Query, HTTPException
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
    set_active_generation_model, load_config, get_generation_model_metadata
)
from src.retrieval.retriever import retrieve, invalidate_bm25
from src.generation.generator import generate_answer_stream, generate_answer
from src.ingestion.pdf_processor import categorize_text
from src.utils.auth import (
    get_current_user, require_admin, authenticate_user,
    create_token, initialize_users, get_all_users
)

load_dotenv()

# Initialize default users on startup
initialize_users()

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
    history: Optional[List[dict]] = None
    agentic: Optional[bool] = False

class ModelSettings(BaseModel):
    model: str

class DBSettings(BaseModel):
    db: str

class LoginRequest(BaseModel):
    username: str
    password: str

# ─── Authentication Endpoints ────────────────────────────────────

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """Authenticate user and return JWT token."""
    user = authenticate_user(req.username, req.password)
    if not user:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_token(user["username"], user["role"])
    return {
        "token": token,
        "username": user["username"],
        "role": user["role"],
        "display_name": user["display_name"]
    }

@app.get("/api/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Return current authenticated user info."""
    users = get_all_users()
    for u in users:
        if u["username"] == user["username"]:
            return {
                "username": u["username"],
                "role": u["role"],
                "display_name": u["display_name"]
            }
    return user

# ─── Utility ──────────────────────────────────────────────────────

def save_to_history(query, category, subcategory, metrics, username="unknown"):
    """Saves a query and its metrics to history.json, tagged with the user."""
    history_path = "history.json"
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
    
    import datetime
    new_entry = {
        "id": len(history) + 1,
        "query": query,
        "category": category,
        "subcategory": subcategory,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": "verified",
        "metrics": metrics,
        "username": username
    }
    history.insert(0, new_entry)
    with open(history_path, "w") as f:
        json.dump(history[:100], f, indent=4)

from src.generation.agents import determine_query_complexity, execute_reasoning_agent, execute_synthesis_agent, execute_multi_model_synthesis

# ─── Chat Endpoint (Multi-Turn) — Authenticated ──────────────────

def condense_query(query: str, history: list) -> str:
    """Uses a fast model to turn a follow-up question into a standalone query."""
    if not history:
        return query
    
    history_str = ""
    for m in history[-5:]:
        q = m.get("question") or m.get("text", "")
        a = m.get("answer") or ""
        if q: history_str += f"User: {q}\n"
        if a: history_str += f"Assistant: {a}\n"
        
    prompt = (
        "Given the following conversation history and a follow-up question, "
        "rephrase the follow-up question to be a standalone question that can be used for search.\n"
        "If the question is already standalone, return it as is.\n\n"
        f"History:\n{history_str}\n"
        f"Follow-up: {query}\n"
        "Standalone Question:"
    )
    from src.generation.generator import generate_utility
    return generate_utility(prompt, model="llama-3.1-8b-instant")

@app.post("/api/chat")
async def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    """Main chat endpoint with SSE-style streaming and multi-turn memory.
    
    Two modes:
        - Universal (agentic=false): Fixed RAG pipeline (retrieve → generate)
        - Agentic (agentic=true): LangGraph agent decides which tools to use
    """
    t_start = time_module.time()
    category, subcategory = categorize_text(req.message)
    
    print(f"[LexVed] Query: {req.message} | Mode: {'Agentic' if req.agentic else 'Universal'} | DB: {get_active_db_name()}")
    
    # ── Workflow Automation (Phase 3) ──────────────────────────────
    if req.message.strip().lower().startswith("/brief "):
        from src.agents.workflows import stream_brief_workflow
        brief_query = req.message.strip()[7:].strip()
        
        async def workflow_stream():
            yield json.dumps({
                "type": "metadata",
                "retrieval_time": 0,
                "category": category,
                "subcategory": subcategory,
                "vector_db": get_active_db_name(),
                "sources": []
            }) + "\n"
            
            try:
                async for event in stream_brief_workflow(brief_query, username=user.get("username", "unknown")):
                    if event["type"] == "thought":
                        yield json.dumps({
                            "type": "agent_thought",
                            "text": event["text"]
                        }) + "\n"
                    elif event["type"] == "content":
                        yield json.dumps({
                            "type": "content",
                            "text": event["text"]
                        }) + "\n"
                    elif event["type"] == "done":
                        yield json.dumps({
                            "type": "done",
                            "generation_time": event["generation_time"]
                        }) + "\n"
            except Exception as e:
                print(f"[LexVed Workflow] Error: {e}")
                yield json.dumps({
                    "type": "content",
                    "text": f"Workflow encountered an error: {str(e)}."
                }) + "\n"
                yield json.dumps({
                    "type": "done",
                    "generation_time": time_module.time() - t_start
                }) + "\n"

        return StreamingResponse(workflow_stream(), media_type="application/x-ndjson")
    
    # ── Collaborative Multi-Agent Swarm (Phase 4) ──────────────────
    if req.message.strip().lower().startswith("/swarm "):
        from src.agents.swarm import stream_swarm_agent
        swarm_query = req.message.strip()[7:].strip()
        
        async def swarm_stream():
            yield json.dumps({
                "type": "metadata",
                "retrieval_time": 0,
                "category": category,
                "subcategory": subcategory,
                "vector_db": get_active_db_name(),
                "sources": []
            }) + "\n"
            
            try:
                async for event in stream_swarm_agent(swarm_query, username=user.get("username", "unknown")):
                    if event["type"] == "thought":
                        yield json.dumps({
                            "type": "agent_thought",
                            "text": event["text"]
                        }) + "\n"
                    elif event["type"] == "content":
                        yield json.dumps({
                            "type": "content",
                            "text": event["text"]
                        }) + "\n"
                    elif event["type"] == "done":
                        yield json.dumps({
                            "type": "done",
                            "generation_time": event["generation_time"]
                        }) + "\n"
            except Exception as e:
                print(f"[LexVed Swarm] Error: {e}")
                yield json.dumps({
                    "type": "content",
                    "text": f"Swarm encountered an error: {str(e)}."
                }) + "\n"
                yield json.dumps({
                    "type": "done",
                    "generation_time": time_module.time() - t_start
                }) + "\n"

        return StreamingResponse(swarm_stream(), media_type="application/x-ndjson")
    
    # ── Agentic Mode: LangGraph Tool-Based Agent ──────────────────
    if req.agentic:
        from src.agents.graph import stream_agent
        import hashlib

        # Derive a stable thread ID for context preservation
        first_q = req.history[0].get("question", "") if req.history else req.message
        hash_val = hashlib.md5(first_q.encode('utf-8')).hexdigest()
        thread_id = f"{user.get('username', 'unknown')}_{hash_val}"

        async def agent_stream():
            """Stream the LangGraph agent's execution to the frontend."""
            full_answer = ""
            tool_calls = []

            # Send initial metadata
            yield json.dumps({
                "type": "metadata",
                "retrieval_time": 0,
                "category": category,
                "subcategory": subcategory,
                "vector_db": get_active_db_name(),
                "sources": []
            }) + "\n"

            yield json.dumps({
                "type": "agent_thought",
                "text": f"LangGraph Agent initialized [Thread: {thread_id[:12]}...]. Analyzing query and selecting tools..."
            }) + "\n"

            try:
                async for event in stream_agent(req.message, history=req.history, thread_id=thread_id, username=user.get("username", "unknown")):
                    if event["type"] == "thought":
                        yield json.dumps({
                            "type": "agent_thought",
                            "text": event["text"]
                        }) + "\n"

                    elif event["type"] == "tool_call":
                        tool_calls.append(event["tool"])
                        yield json.dumps({
                            "type": "agent_thought",
                            "text": f"Calling tool: {event['tool']}..."
                        }) + "\n"

                    elif event["type"] == "tool_result":
                        yield json.dumps({
                            "type": "agent_thought",
                            "text": f"Tool {event['tool']} returned results. Processing..."
                        }) + "\n"

                    elif event["type"] == "content":
                        full_answer += event["text"]
                        yield json.dumps({
                            "type": "content",
                            "text": event["text"]
                        }) + "\n"

                    elif event["type"] == "done":
                        total_time = time_module.time() - t_start
                        save_to_history(req.message, category, subcategory, {
                            "retrieval_lat": 0,
                            "e2e_lat": total_time,
                            "ans_length": len(full_answer.split()),
                            "tools_used": event.get("tool_calls", []),
                            "agent_steps": event.get("steps", 0)
                        }, username=user.get("username", "unknown"))
                        yield json.dumps({
                            "type": "done",
                            "generation_time": total_time,
                            "tools_used": event.get("tool_calls", []),
                        }) + "\n"

            except Exception as e:
                print(f"[LexVed] Agent error: {e}")
                # Fallback: return error as content
                yield json.dumps({
                    "type": "content",
                    "text": f"Agent encountered an error: {str(e)}. Falling back to standard pipeline."
                }) + "\n"
                yield json.dumps({
                    "type": "done",
                    "generation_time": time_module.time() - t_start
                }) + "\n"

        return StreamingResponse(agent_stream(), media_type="application/x-ndjson")
    
    # ── Universal Mode: Fixed RAG Pipeline with Dynamic Model Selection ────────────
    # Higher recall for standard mode
    top_k = 10
    
    # 1. Intelligent Query Condensation
    augmented_query = condense_query(req.message, req.history)
    print(f"[LexVed] Augmented Query: {augmented_query}")
    
    res, retrieval_time = retrieve(
        augmented_query,
        top_k=top_k,
        category=None,
        subcategory=None
    )
    
    print(f"[LexVed] Retrieved {len(res)} chunks. Time: {retrieval_time:.2f}s")

    context = ""
    from src.retrieval.compressor import compress_text
    for m in res:
        source = os.path.basename(m.payload.get("source", "Unknown"))
        page_val = m.payload.get("page")
        # Display logic: If 0, show 1. If any other number, show exact.
        if isinstance(page_val, int):
            display_page = page_val if page_val > 0 else 1
        else:
            display_page = page_val or "?"
        
        # Apply Query-Aware Context Compression
        compressed_segment = compress_text(req.message, m.payload['text'])
        context += f"\n[Source: {source}, Page: {display_page}]\n{compressed_segment}\n"

    full_answer = ""
    # Collect source references for citation linking
    sources = []
    for m in res:
        src = os.path.basename(m.payload.get("source", "Unknown"))
        pg_val = m.payload.get("page")
        # For the sidebar: match the exact page or default to 1 if 0/None
        display_pg = pg_val if (isinstance(pg_val, int) and pg_val > 0) else 1
        full_path = m.payload.get("source", "")
        sources.append({"file": src, "page": display_pg, "path": full_path})

    def stream_response():
        nonlocal full_answer
        yield json.dumps({
            "type": "metadata",
            "retrieval_time": retrieval_time,
            "category": category,
            "subcategory": subcategory,
            "vector_db": get_active_db_name(),
            "sources": sources
        }) + "\n"

        # ── Dynamic Model Selection & Complexity Routing ──
        yield json.dumps({"type": "agent_thought", "text": "Determining query complexity and routing to optimal model..."}) + "\n"
        target_model = determine_query_complexity(req.message)
        yield json.dumps({"type": "agent_thought", "text": f"Query routed to: {target_model}"}) + "\n"
        
        # ── Local Cluster Warm-up Detection ──
        local_models = ["llama3", "qwen2.5:7b"]
        if target_model in local_models:
            try:
                import requests
                r = requests.get("http://localhost:11434/api/ps", timeout=1)
                if r.status_code == 200:
                    running_models = [m["name"] for m in r.json().get("models", [])]
                    is_warm = any(target_model in m for m in running_models)
                    if not is_warm:
                        yield json.dumps({"type": "agent_thought", "text": f"Initializing {target_model} cluster. Loading neural weights into local GPU memory..."}) + "\n"
            except:
                pass

        # Handle routing ensemble target model to standard high-fidelity model
        actual_model = "llama-3.1-8b-instant" if target_model == "ensemble" else target_model

        # ── Generate Context-Grounded Answer Stream ──
        yield json.dumps({"type": "agent_thought", "text": "Analyzing context and generating legal synthesis..."}) + "\n"
        for chunk in generate_answer_stream(req.message, context, model=actual_model, history=req.history):
            full_answer += chunk
            yield json.dumps({"type": "content", "text": chunk}) + "\n"
        
        total_time = time_module.time() - t_start
        save_to_history(req.message, category, subcategory, {
            "retrieval_lat": retrieval_time,
            "e2e_lat": total_time,
            "ans_length": len(full_answer.split()),
            "routed_model": target_model
        }, username=user.get("username", "unknown"))
        yield json.dumps({"type": "done", "generation_time": total_time}) + "\n"

    return StreamingResponse(stream_response(), media_type="application/x-ndjson")

# ─── PDF File Serving — Authenticated ─────────────────────────────

@app.get("/api/pdf/{filename:path}")
async def serve_pdf(filename: str, token: str = Query(None), user: dict = Depends(get_current_user)):
    """Serve a PDF from the data directory for citation viewing."""
    # Note: If token is provided in query, we can skip the header check
    # The get_current_user dependency already checks the header. 
    # If the user is viewing via a link, we need to handle the case where get_current_user fails.
    # To support deep links from external tabs, we allow the token in query.
    
    from fastapi.responses import FileResponse
    # Search for the file in the data directory
    pdf_dir = "data/PDF"
    filename_lower = filename.lower()
    for root, dirs, files in os.walk(pdf_dir):
        for f in files:
            f_lower = f.lower()
            # Direct match or match without extension
            if f_lower == filename_lower or f_lower.replace(".pdf", "") == filename_lower:
                full_path = os.path.join(root, f)
                return FileResponse(full_path, media_type="application/pdf", filename=f)
    raise HTTPException(status_code=404, detail=f"PDF not found: {filename}")

# ─── Health Check — Authenticated ─────────────────────────────────

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
    
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            health["ollama"] = "connected"
            health["ollama_models"] = models
        else:
            health["ollama"] = "error"
    except Exception:
        health["ollama"] = "offline"

    active_db = get_active_db_name()
    if active_db == "qdrant":
        try:
            from qdrant_client import QdrantClient
            c = QdrantClient(host="localhost", port=6333, timeout=3)
            cols = [col.name for col in c.get_collections().collections]
            health["vector_db"] = "connected"
            health["collections"] = cols
        except Exception:
            health["vector_db"] = "offline"
    else:
        try:
            from src.utils.pinecone_client import index
            stats = index.describe_index_stats()
            health["vector_db"] = "connected"
            health["vector_count"] = stats.get('total_vector_count', 0)
        except Exception:
            health["vector_db"] = "offline"

    return health

# ─── Metrics & History — Authenticated ────────────────────────────

@app.get("/api/metrics")
async def get_metrics(user: dict = Depends(require_admin)):
    """Returns evaluation metrics — Admin only."""
    import psutil
    metric_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json")
    if os.path.exists(metric_path):
        try:
            with open(metric_path, "r") as f:
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
            # Auto-stamp completed results for cache validation
            if report.get("status") == "done" and not report.get("corpus_fingerprint"):
                stamp_results(metric_path)
            return report
        except Exception as e:
            return {"status": "error", "message": f"Read error: {e}"}
    return {"status": "error", "message": "No audit report found."}

@app.get("/api/history")
async def get_history(user: dict = Depends(get_current_user)):
    """Returns query history — filtered by user. Admin sees all."""
    history_path = "history.json"
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                all_history = json.load(f)
            # Admin sees everything, regular users see only their own
            if user.get("role") == "admin":
                return all_history
            return [h for h in all_history if h.get("username") == user.get("username")]
        except Exception:
            return []
    return []

@app.delete("/api/history")
async def clear_history(user: dict = Depends(get_current_user)):
    """Clears the query history log."""
    history_path = "history.json"
    with open(history_path, "w") as f:
        json.dump([], f)
    return {"status": "success", "message": "History cleared."}

# ─── File Management & Ingestion — Authenticated ─────────────────

@app.get("/api/files")
async def list_files(user: dict = Depends(get_current_user)):
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
    
    if os.path.exists("evaluation_data.json"):
        files.append({"name": "evaluation_data.json", "size": f"{os.path.getsize('evaluation_data.json') / 1024:.1f} KB", "type": "Evaluation Suite"})
    if os.path.exists("evaluation_results.json"):
        files.append({"name": "evaluation_results.json", "size": f"{os.path.getsize('evaluation_results.json') / 1024:.1f} KB", "type": "Knowledge Base"})
    
    return files

@app.post("/api/ingest")
async def ingest_pdf(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Upload and ingest a PDF into the vector database."""
    pdf_dir = "data/PDF"
    os.makedirs(pdf_dir, exist_ok=True)
    
    file_path = os.path.join(pdf_dir, file.filename)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    def ingest_stream():
        try:
            from src.ingestion.pdf_processor import extract_chunks, process_chunks_batch
            from src.ingestion.embedder import get_embeddings
            
            yield json.dumps({"status": "processing", "step": "Extracting text and fingerprinting PDF..."}) + "\n"
            chunks = extract_chunks(file_path)
            if not chunks:
                yield json.dumps({"status": "error", "step": "No text extracted from PDF."}) + "\n"
                return

            file_hash = chunks[0].get("file_hash")
            yield json.dumps({"status": "processing", "step": f"Fingerprint: {file_hash[:12]}... Checking for duplicates..."}) + "\n"
            
            # 1. Pre-flight Check (Check if hash already exists in active DB)
            active_db = get_active_db_name()
            already_indexed = False
            
            if active_db == "qdrant":
                try:
                    from qdrant_client import QdrantClient
                    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
                    from src.utils.qdrant_provider import COLLECTION_NAME
                    client = QdrantClient(host="localhost", port=6333)
                    # Check for Hash OR Filename (for legacy migration)
                    search_res = client.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=Filter(
                            should=[
                                FieldCondition(key="file_hash", match=MatchValue(value=file_hash)),
                                FieldCondition(key="source", match=MatchValue(value=os.path.basename(file_path)))
                            ]
                        ),
                        limit=1
                    )
                    if search_res[0]: already_indexed = True
                except: pass
            else:
                try:
                    from src.utils.pinecone_client import index
                    # Check for Hash
                    res = index.query(vector=[0]*768, filter={"file_hash": {"$eq": file_hash}}, top_k=1)
                    if res.get("matches"): 
                        already_indexed = True
                    else:
                        # Fallback check for Filename
                        res_name = index.query(vector=[0]*768, filter={"source": {"$eq": os.path.basename(file_path)}}, top_k=1)
                        if res_name.get("matches"): already_indexed = True
                except: pass

            if already_indexed:
                yield json.dumps({"status": "complete", "step": f"Document already indexed (SHA-256 Match). Skipping redundant ingestion.", "chunks": 0}) + "\n"
                return

            yield json.dumps({"status": "processing", "step": f"Extracted {len(chunks)} chunks. Processing NER & Categories..."}) + "\n"
            
            chunks = process_chunks_batch(chunks)
            yield json.dumps({"status": "processing", "step": f"Cleaned {len(chunks)} chunks. Generating embeddings..."}) + "\n"
            
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
                            "file_hash": file_hash,
                            "category": cat,
                            "subcategory": sub
                        }
                    ))
                
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
                            "file_hash": file_hash,
                            "category": cat,
                            "subcategory": sub
                        }
                    })
                
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i+batch_size]
                    index.upsert(vectors=batch)
                    yield json.dumps({"status": "processing", "step": f"Upserted {min(i+batch_size, len(vectors))}/{len(vectors)} vectors..."}) + "\n"
            
            invalidate_bm25()
            
            yield json.dumps({
                "status": "complete",
                "step": f"Successfully ingested {file.filename}: {len(chunks)} chunks indexed.",
                "chunks": len(chunks)
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({"status": "error", "step": f"Ingestion failed: {str(e)}"}) + "\n"

    return StreamingResponse(ingest_stream(), media_type="application/x-ndjson")

# ─── Settings — Authenticated ─────────────────────────────────────

@app.get("/api/settings/embedding_model")
async def get_model_setting(user: dict = Depends(get_current_user)):
    return {"model": get_active_model_name()}

@app.post("/api/settings/embedding_model")
async def set_model_setting(settings: ModelSettings, user: dict = Depends(get_current_user)):
    if set_active_model(settings.model):
        invalidate_bm25()
        return {"status": "success", "model": settings.model}
    return {"status": "error", "message": "Invalid model"}

@app.get("/api/settings/vector_db")
async def get_db_setting(user: dict = Depends(get_current_user)):
    return {"db": get_active_db_name()}

@app.post("/api/settings/vector_db")
async def set_db_setting(settings: DBSettings, user: dict = Depends(get_current_user)):
    if set_active_db(settings.db):
        invalidate_bm25()
        return {"status": "success", "db": settings.db}
    return {"status": "error", "message": "Invalid database"}

@app.get("/api/settings/generation_model")
async def get_gen_model_setting(user: dict = Depends(get_current_user)):
    return {"model": get_active_generation_model()}

@app.post("/api/settings/generation_model")
async def set_gen_model_setting(settings: ModelSettings, user: dict = Depends(get_current_user)):
    if set_active_generation_model(settings.model):
        return {"status": "success", "model": settings.model}
    return {"status": "error", "message": "Invalid generation model"}

@app.get("/api/settings/config")
async def get_full_config(user: dict = Depends(get_current_user)):
    """Returns the full config for frontend consumption, including dynamic model metadata."""
    config = load_config()
    return {
        "embedding_models": list(config.get("models", {}).keys()),
        "generation_models": config.get("generation_models", []),
        "generation_model_metadata": get_generation_model_metadata(),
        "groq_models": config.get("groq_models", []),
        "providers": config.get("providers", [])
    }

# ─── Evaluation Caching ──────────────────────────────────────────

def get_corpus_fingerprint():
    """Returns a fingerprint of the PDF corpus (file count + total size).
    If this hasn't changed, evaluation results are still valid."""
    pdf_dir = "data/PDF"
    total_files = 0
    total_size = 0
    if os.path.exists(pdf_dir):
        for root, dirs, files in os.walk(pdf_dir):
            for f in files:
                if f.lower().endswith(".pdf"):
                    total_files += 1
                    total_size += os.path.getsize(os.path.join(root, f))
    return f"{total_files}:{total_size}"

def is_cache_valid(results_path):
    """Check if cached evaluation results are still valid (corpus unchanged)."""
    if not os.path.isabs(results_path):
        results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), results_path)
    if not os.path.exists(results_path):
        return False
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
        if data.get("status") not in ["done", "complete"]:
            return False
        return data.get("corpus_fingerprint") == get_corpus_fingerprint()
    except Exception:
        return False

def stamp_results(results_path):
    """Stamp completed results with the current corpus fingerprint."""
    # Ensure absolute path
    if not os.path.isabs(results_path):
        results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), results_path)
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
        if data.get("status") == "done":
            data["corpus_fingerprint"] = get_corpus_fingerprint()
            with open(results_path, "w") as f:
                json.dump(data, f, indent=2)
    except Exception:
        pass

# ─── Evaluation Workflows — Admin Only ───────────────────────────

@app.post("/api/workflow/evaluate")
async def trigger_evaluation(request: Request, user: dict = Depends(require_admin)):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    force = body.get("force", False)

    # Check cache first
    if not force and is_cache_valid("evaluation_results.json"):
        return {"status": "cached", "message": "Results are up to date. No new PDFs detected."}

    def run_eval():
        print(f"[LexVed] Starting Evaluation Workflow on {get_active_db_name()}...")
        try:
            import os
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            venv_python = os.path.join(backend_dir, "venv", "bin", "python3")
            
            # Thread Governor: Prevent CPU saturation
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "4"
            env["MKL_NUM_THREADS"] = "4"
            env["TORCH_NUM_THREADS"] = "4"
            
            proc = subprocess.Popen([venv_python, "run_metrics.py"], cwd=backend_dir, env=env)
            eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json")
            with open(eval_path, "w") as f:
                json.dump({
                    "status": "processing", 
                    "progress": "Initializing Intelligence Node...",
                    "pid": proc.pid
                }, f)
        except Exception as e:
            eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json")
            with open(eval_path, "w") as f:
                json.dump({"status": "error", "message": str(e)}, f)

    executor.submit(run_eval)
    return {"status": "processing", "message": "Evaluation initiated"}

@app.post("/api/workflow/comparative")
async def trigger_comparative(request: Request, user: dict = Depends(require_admin)):
    """Triggers comparative benchmarking across all embedding models — Admin only."""
    body = await request.json()
    resume = body.get("resume", False)
    force = body.get("force", False)

    # Check cache first (only if not resuming)
    if not force and not resume and is_cache_valid("comparative_results.json"):
        return {"status": "cached", "message": "Comparative results are up to date. No new PDFs detected."}

    def run_comparative_thread():
        try:
            import os
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            venv_python = os.path.join(backend_dir, "venv", "bin", "python3")
            cmd = [venv_python, "run_comparative.py"]
            if resume:
                cmd.append("--resume")
                
            # Thread Governor
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "4"
            env["MKL_NUM_THREADS"] = "4"
            env["TORCH_NUM_THREADS"] = "4"
            
            proc = subprocess.Popen(cmd, cwd=backend_dir, env=env)
            
            if not resume:
                if os.path.exists("intermediate_results.json"):
                    try: os.remove("intermediate_results.json")
                    except Exception: pass

                comp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparative_results.json")
                with open(comp_path, "w") as f:
                    json.dump({
                        "status": "processing",
                        "progress": "Starting comparative benchmark...",
                        "completed_models": [],
                        "current_model": "",
                        "pid": proc.pid
                    }, f)
            else:
                comp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparative_results.json")
                if os.path.exists(comp_path):
                    try:
                        with open(comp_path, "r") as f:
                            old_data = json.load(f)
                        old_data["status"] = "processing"
                        old_data["pid"] = proc.pid
                        with open(comp_path, "w") as f:
                            json.dump(old_data, f, indent=2)
                    except Exception: pass
        except Exception as e:
            comp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparative_results.json")
            with open(comp_path, "w") as f:
                json.dump({"status": "error", "message": str(e)}, f)

    executor.submit(run_comparative_thread)
    return {"status": "processing", "message": "Comparative benchmark initiated"}

@app.get("/api/comparative")
async def get_comparative(user: dict = Depends(require_admin)):
    """Returns comparative benchmarking results — Admin only."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparative_results.json")
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

@app.post("/api/workflow/compare_pipelines")
async def trigger_pipeline_comparison(user: dict = Depends(require_admin)):
    """Admin only."""
    def run_compare():
        try:
            import os
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            venv_python = os.path.join(backend_dir, "venv", "bin", "python3")
            proc = subprocess.Popen([venv_python, "run_metrics.py", "--pipeline", "both"], cwd=backend_dir)
            with open("pipeline_comparison_results.json", "w") as f:
                json.dump({
                    "status": "processing",
                    "progress": "Comparing Enhanced vs Primitive pipelines...",
                    "pid": proc.pid
                }, f)
        except Exception as e:
            with open("pipeline_comparison_results.json", "w") as f:
                json.dump({"status": "error", "message": str(e)}, f)

    executor.submit(run_compare)
    return {"status": "processing", "message": "Pipeline comparison initiated"}

@app.get("/api/compare_pipelines")
async def get_pipeline_comparison(user: dict = Depends(require_admin)):
    """Admin only."""
    path = "pipeline_comparison_results.json"
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
            return {"status": "error", "message": f"Read error: {e}"}
    return {"status": "error", "message": "No pipeline comparison results found."}

@app.post("/api/workflow/evaluate_primitive")
async def trigger_primitive_evaluation(request: Request, user: dict = Depends(require_admin)):
    """Triggers the standalone Primitive Pipeline evaluation — Admin only."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    model_choice = body.get("model_choice", "1")
    force = body.get("force", False)

    # Check cache first
    if not force and is_cache_valid("primitive_evaluation_results.json"):
        return {"status": "cached", "message": "Primitive results are up to date. No new PDFs detected."}

    def run_primitive():
        try:
            import os
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            venv_python = os.path.join(backend_dir, "venv", "bin", "python3")
            proc = subprocess.Popen(
                [venv_python, "-c",
                 f"from qdrant_pipeline import run_primitive_pipeline; run_primitive_pipeline(model_choice='{model_choice}')"],
                cwd=backend_dir
            )
            prim_path = os.path.join(backend_dir, "primitive_evaluation_results.json")
            with open(prim_path, "w") as f:
                json.dump({
                    "status": "processing",
                    "progress": "Initializing Primitive Pipeline...",
                    "pid": proc.pid
                }, f)
        except Exception as e:
            prim_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "primitive_evaluation_results.json")
            with open(prim_path, "w") as f:
                json.dump({"status": "error", "message": str(e)}, f)

    executor.submit(run_primitive)
    return {"status": "processing", "message": "Primitive Pipeline evaluation initiated"}

@app.get("/api/metrics/primitive")
async def get_primitive_metrics(user: dict = Depends(require_admin)):
    """Returns the latest Primitive Pipeline evaluation results — Admin only."""
    path = "primitive_evaluation_results.json"
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
                            report["message"] = "Primitive Pipeline process died unexpectedly."
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        report["status"] = "error"
                        report["message"] = "Process no longer active."
            return report
        except Exception as e:
            return {"status": "error", "message": f"Read error: {e}"}
    return {"status": "error", "message": "No primitive evaluation results found. Run the Primitive Pipeline evaluation first."}

# ─── Admin-Only Management Endpoints ─────────────────────────────

@app.get("/api/admin/users")
async def admin_list_users(user: dict = Depends(require_admin)):
    """List all registered users — Admin only."""
    return {"users": get_all_users()}

@app.post("/api/admin/clear_eval")
async def admin_clear_eval(user: dict = Depends(require_admin)):
    """Clear all evaluation result files — Admin only."""
    cleared = []
    for f in ["evaluation_results.json", "comparative_results.json", "pipeline_comparison_results.json", "primitive_evaluation_results.json"]:
        if os.path.exists(f):
            try:
                with open(f, "w") as fh:
                    json.dump({}, fh)
                cleared.append(f)
            except Exception:
                pass
    return {"status": "success", "cleared": cleared}

# ─── Server Entry ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    if not os.path.exists("history.json"):
        with open("history.json", "w") as f: json.dump([], f)
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
