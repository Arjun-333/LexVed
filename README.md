# LexVed: Advanced Legal Research RAG Platform

LexVed is a professional-grade legal RAG system powered exclusively by **Llama 3 8B** running locally via Ollama. All inference is private and offline.

## Key Features
- **Hierarchical Sub-indexing:** Criminal and Civil case law partitioned for precision
- **Llama 3 (Local):** All generation runs on your machine via Ollama. No cloud APIs, no data leaves your system
- **Turbo Ingestion:** Parallel PDF processing with SpaCy NER for 500+ documents
- **PII Scrubbing:** Automatic redaction of Names, Aadhaar, PAN, and Emails
- **Obsidian Gavel UI:** Premium glassmorphic dashboard with citation-aware responses

## Quick Start

### 1. Start Qdrant
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Pull Llama 3
```bash
ollama pull llama3
```

### 3. Ingest Documents
Place PDFs in `backend/data/PDF/CRIMINAL/` or `backend/data/PDF/CIVIL/`, then:
```bash
cd backend && ./venv/bin/python3 test_embedding_qdrant.py
```

### 4. Start the Platform
```bash
./venv/bin/python3 app.py
```

### 5. Open LexVed
Visit **http://localhost:5000**

## Architecture
- **Vector DB:** Qdrant (Self-hosted)
- **LLM:** Llama 3 8B via Ollama (localhost:11434)
- **Embeddings:** Sentence-Transformers (multi-qa-mpnet-base-cos-v1)
- **NER:** SpaCy (en_core_web_sm)

For detailed engineering design, see **[DPR.md](DPR.md)**.

---
**LexVed | Private Legal Intelligence**
