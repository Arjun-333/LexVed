# LexVed Command Reference

This document provides a technical reference for the essential commands required to manage, ingest, and benchmark the LexVed RAG platform.

## Core Services

### 1. Start Backend Server
```bash
cd backend && source venv/bin/activate
python3 app.py
```
**Description:** Initializes the FastAPI backend and AI agents. This service must be active to enable the API for the frontend and benchmarking suite.

### 2. Start Frontend UI
```bash
cd frontend
npm run dev
```
**Description:** Launches the Next.js frontend dashboard. Accessible by default at `http://localhost:3000`.

### 3. Start Local Vector Database (Qdrant)
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```
**Description:** Launches the Qdrant vector database via Docker. Required for local low-latency retrieval testing.

---

## Data Ingestion and Indexing

### 4. Full Corpus Ingestion (Pinecone)
```bash
cd backend && source venv/bin/activate
python3 ingest_pinecone.py
```
**Description:** Processes the legal corpus of 516 PDFs and uploads the initial 19,500 chunks to the Pinecone vector database using the MPNet embedding model.

### 5. Hierarchical Indexing Validation
```bash
cd backend && source venv/bin/activate
python3 test_subindexing.py
```
**Description:** Validates the domain-isolation logic for hierarchical sub-indexing. Ensures that queries are correctly routed to specific legal sub-domains (Civil vs. Criminal).

---

## Benchmarking and Auditing

### 6. Run Pinecone Benchmark (MPNet)
```bash
cd backend && source venv/bin/activate
python3 run_primitive_all_models.py
```
**Description:** Executes a comprehensive 24-KPI audit of the Pinecone backend against the standardized 10-query legal benchmark. Generates a standalone Pinecone analysis report in PDF format.

### 7. Run Qdrant Benchmark (MPNet)
```bash
cd backend && source venv/bin/activate
python3 run_qdrant_all_models.py
```
**Description:** Executes the audit suite against the local Qdrant backend. Note: The initial execution requires approximately 2 hours for local embedding generation.

### 8. Run Enhanced vs. Baseline Comparison
```bash
cd backend && source venv/bin/activate
python3 run_comparative.py
```
**Description:** Directly compares the performance of the Multi-Agent Enhanced Pipeline against the Baseline Primitive Pipeline.

### 9. Generate Final Comparative Audit Report
```bash
cd backend && source venv/bin/activate
python3 generate_final_report.py
```
**Description:** Synthesizes results from both Pinecone and Qdrant benchmarks into a single high-fidelity comparative report, providing detailed architectural analysis.

---

## Diagnostics and Maintenance

### 10. System Health Diagnostic
```bash
cd backend && source venv/bin/activate
python3 diagnostic.py
```
**Description:** Performs an end-to-end check of all neural engines, vector repositories, and local inference nodes (Ollama/Llama 3).

### 11. Monitor Background Process Logs
```bash
# Monitor Pinecone benchmark progress
tail -f backend/pinecone_run.log

# Monitor Qdrant benchmark progress
tail -f backend/qdrant_run.log
```
**Description:** Real-time monitoring of ongoing background benchmarking and ingestion tasks.

### 12. Verify Data Integrity (Duplicate Check)
```bash
python3 scratch/test_duplicate_logic.py
```
**Description:** Verifies that the SHA-256 fingerprinting logic correctly identifies and skips duplicate PDFs.

---

## Service Termination

### 13. Stop All Services (Port Release)
```bash
# Terminate Backend (Port 5000)
fuser -k 5000/tcp

# Terminate Frontend (Port 3000)
fuser -k 3000/tcp

### 14. Kill All LexVed Services (One-Liner)
```bash
fuser -k 5000/tcp && fuser -k 3000/tcp
```
**Description:** Forcefully terminates both the backend and frontend services in a single command. Recommended for rapid environment resets.
```
**Description:** Forcefully terminates any processes listening on the backend or frontend ports. Use this if you encounter "Address already in use" errors during startup.

