# LexVed: Detailed Project Report

## Project Vision
LexVed is a professional-grade legal RAG (Retrieval-Augmented Generation) system. It uses **Hierarchical Sub-indexing** to partition legal documents into distinct domains (Criminal, Civil), ensuring noise-free, precise legal research.

LexVed runs exclusively on **Llama 3 8B (via Ollama)** for complete privacy. No data ever leaves your machine.

---

## Technical Design Decisions

### 1. Hierarchical Sub-indexing (Precision Engine)
Legal terms often overlap across domains. LexVed partitions data using Qdrant payloads to ensure domain isolation.
- **Mapping:** Data is categorized based on its origin folder (CRIMINAL vs. CIVIL).
- **Sub-sorting:** Results are sorted by semantic relevance, then by page number for narrative coherence.

### 2. Llama 3 Local Inference (Privacy-First)
All generation runs locally on your machine via Ollama.
- **Model:** `llama3` (8B parameters)
- **Endpoint:** `localhost:11434`
- **Benefit:** Zero cloud dependency, full document privacy, works offline.

### 3. Turbo Ingestion Engine (Parallel Scaling)
- **Parallelism:** Distributed PDF processing across multi-core CPUs.
- **SpaCy NER:** Free, open-source, 10x faster than BERT for name redaction.
- **Incremental Indexing:** Tracks processed files in `ingested_files.json`.

### 4. PII Redaction & Privacy
- **ML Layer:** SpaCy `en_core_web_sm` for entity detection.
- **Regex Layer:** Scrubs Aadhaar, PAN, Emails, and Phone Numbers.
- **Result:** Documents are sanitized locally before any text reaches the LLM.

### 5. Obsidian Gavel UI
The frontend is a premium **Glassmorphic** dashboard:
- **HSL Design System:** Vantablack and Legal Gold palette.
- **Citation Cards:** Responses include dedicated citation boxes with page-level links.
- **Provider Badge:** Shows "Llama 3 8B - Local" to confirm private inference.

---

## System Architecture

### Ingestion Flow
1. **Extraction:** PDF text extracted using `PyMuPDF`.
2. **Parallel NER:** SpaCy detects and redacts personal names.
3. **Embedding:** `multi-qa-mpnet-base-cos-v1` converts text into 768-dim vectors.
4. **Persistence:** Vectors stored in Qdrant with `category` filtering.

### Retrieval Flow
1. **Intent-Aware Retrieval:** Detects Criminal/Civil intent and applies sub-index filter.
2. **Generation:** Llama 3 generates the answer with strict citation rules.
3. **Citation Assembly:** Context injected with source/page markers.

---

## Command Reference

### Starting the System
1. **Start Qdrant:** `docker run -p 6333:6333 qdrant/qdrant`
2. **Pull Llama 3:** `ollama pull llama3`
3. **Ingest PDFs:** `cd backend && ./venv/bin/python3 test_embedding_qdrant.py`
4. **Start Platform:** `./venv/bin/python3 app.py`
5. **Open UI:** Visit `http://localhost:5000`

---
**LexVed | Private Legal Intelligence**
