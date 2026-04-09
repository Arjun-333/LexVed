# LexVed Document Ingestion and Retrieval System

### 5. Hybrid Search (Keyword + Vector)
**How:** Combines Qdrant's vector similarity with a `match_text` filter for exact legal keywords.
**Why:** Legal research often depends on specific statute numbers (e.g., "302"). Hybrid search ensures these terms are boosted while maintaining the semantic depth of the vector search.

### 6. Page-Aware Sorting
**How:** Results with similar semantic scores are sub-sorted by their page number.
**Why:** Improves the narrative coherence of the retrieved context, presenting legal arguments in the order they appear in the original document.

## Project Vision
LexVed is a professional-grade legal RAG (Retrieval-Augmented Generation) system. Unlike standard flat-search pipelines, LexVed is designed with **Hierarchical Sub-indexing**, ensuring that legal documents are partitioned into distinct domains (Criminal, Civil). This ensures that lawyers and legal researchers get the most precise citations without noise from unrelated legal fields.

---

## How and Why: Engineering Design Decisions

### 1. Hierarchical Sub-indexing (Precision Engine)
**The Problem:** In legal research, terms like "theft" can appear in both Criminal cases and Civil insurance disputes. A flat vector search might confuse the two.
**The Solution:** LexVed partitions data at the database level using Qdrant payloads. 
- **Mapping:** Documents in `backend/data/PDF/CRIMINAL/` are tagged as `Criminal`.
- **Isolation:** On query, the system identifies the user's intent (e.g., "crime") and restricts the search scope *exclusively* to the Criminal sub-index.
- **Why:** This eliminates cross-domain noise and ensures the AI context is 100% relevant.

### 2. Payload Indexing (Speed & Scalability)
**The Implementation:** We created explicit Keyword and Integer indices in Qdrant for `category`, `subcategory`, and `page`.
**Why:** Standard vector search is $O(N)$. By indexing these fields, Qdrant performs **Pre-filtering**, narrowing down the search space *before* the expensive cosine similarity math. 

### 3. Automated PII Redaction (Privacy Compliance)
**The Implementation:** A dual-layer system using:
- **ML Layer:** `dslim/bert-base-NER` detects and redacts personal names.
- **Regex Layer:** High-precision patterns scrub Aadhaar, PAN, Phone numbers, and Emails.
**Why:** Privacy is paramount in legal tech. This ensures sensitive lawyer-client information never reaches the cloud LLM.

### 4. Page-Level Citation Logic (Verifiability)
**The Implementation:** Metadata markers `[Source: filename, Page: X]` are injected into every context chunk.
**Why:** A legal assistant is only as good as its citations. By forcing the LLM to cite specific pages, LexVed provides **verifiable evidence** rather than just general answers.

---

## System Architecture

### Ingestion Flow
1. **Extraction:** PDF text is extracted using `PyMuPDF (fitz)`.
2. **Analysis:** The system detects the domain based on the folder (`CRIMINAL` vs `CIVIL`).
3. **Chunking:** Text is split using regex that preserves citation patterns (e.g., "Sec. 302 IPC").
4. **Sanitization:** NER and Regex redaction scrub sensitive data.
5. **Embedding:** `multi-qa-mpnet-base-cos-v1` converts text into 768-dimensional vectors.
6. **Upsert:** Vectors + Metadata are stored in Qdrant with explicit payload indexing.

### Retrieval and Chat Flow
1. **Intent Detection:** The system analyzes the query to detect if it's a Criminal or Civil question.
2. **Filtered retrieval:** Searches Qdrant using the detected `category` filter.
3. **Context Assembly:** Chunks are formatted with source/page markers.
4. **Generation:** Gemini 1.5 Pro generates the final answer with strict citation rules.

---

## Prerequisites and Setup

### 1. Qdrant (Vector Database)
The system requires a running Qdrant instance on port 6333.
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Virtual Environment
Always use a virtual environment to avoid global dependency conflicts.
```bash
# Setup
python3 -m venv backend/venv
source backend/venv/bin/activate

# Dependencies
pip install -r requirements.txt
```

### 3. Configuration
Set your Gemini API key:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

---

## Usage Guide

### Step 1: Ingest Documents
Place your PDFs in:
- `backend/data/PDF/CRIMINAL/`
- `backend/data/PDF/CIVIL/`

Run the ingestion pipeline:
```bash
python backend/test_embedding_qdrant.py
```

### Step 2: Verify the Sub-index
Run the verification script to ensure filtering is working correctly:
```bash
python backend/test_subindexing.py
```

### Step 3: Start the Server
```bash
python backend/app.py
```

---

## Repository Structure
- `backend/app.py`: Main Flask API with intent-aware retrieval.
- `backend/src/ingestion/`: Handles PDF processing, NER, and uploading.
- `backend/src/retrieval/`: Filtered similarity search logic.
- `backend/src/utils/`: Qdrant client and collection configuration.
- `backend/test_subindexing.py`: Unit test for hierarchical filtering.

---

## License
Internal Use Only (LexVed Project).
