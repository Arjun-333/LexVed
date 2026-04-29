# LexVed — Legal AI Research Platform

Production-grade Retrieval-Augmented Generation (RAG) system for legal document analysis. Built with a hybrid retrieval pipeline, 6 embedding models, and a 24-KPI institutional benchmark suite.

## Architecture

```
PDF Upload → SpaCy NER Redaction → Citation-Aware Chunking
    ↓
Embedding (6 models) → Vector DB (Qdrant / Pinecone)
    ↓
Query → Dense Search + BM25 Sparse Search → RRF Fusion → CrossEncoder Rerank
    ↓
Llama 3 8B (Ollama) → Streaming Response
    ↓
Evaluation: ROUGE, BLEU, METEOR, BERTScore, DeBERTa NLI, SpaCy NER, LLM Judge
```

## UI/UX Philosophy: "All Black But Gold"
The LexVed interface has been heavily engineered to deliver a premium, museum-grade aesthetic:
*   **Cinematic Typography:** Utilizing `Playfair Display` and `Cinzel` for sharp, high-authority legal headers with wide character tracking.
*   **Hardware-Accelerated 3D Coverflow:** The embedding model omnitrix is powered by native CSS `scroll-snap` mechanics and an `IntersectionObserver`, guaranteeing buttery-smooth 120FPS carousel dragging without JavaScript physics stuttering.
*   **Gilded Accents:** Minimalist `#000000` pitch-black voids contrasted sharply by `#D4AF37` gold luminous interactions.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Llama 3 8B via Ollama (local) |
| Embeddings | SentenceTransformers, Cohere API |
| Vector DB | Qdrant (self-hosted), Pinecone (cloud) |
| Retrieval | Hybrid BM25 + Dense, RRF, CrossEncoder |
| Backend | FastAPI (Python 3.10) |
| Frontend | Next.js 14, Framer Motion, TailwindCSS |
| Evaluation | HuggingFace evaluate, DeBERTa NLI, SpaCy NER |

## Embedding Models (6)

| Model | Dimensions | Source |
|-------|-----------|--------|
| multi-qa-mpnet-base-cos-v1 | 768 | SentenceTransformers |
| multi-qa-MiniLM-L6-cos-v1 | 384 | SentenceTransformers |
| multi-qa-distilbert-cos-v1 | 768 | SentenceTransformers |
| BAAI/bge-m3 | 1024 | HuggingFace |
| intfloat/multilingual-e5-large-instruct | 1024 | HuggingFace |
| Cohere embed-english-v3.0 | 1024 | Cohere API |

## Prerequisites

- Python 3.10+
- Node.js 18+
- Ollama with Llama 3 pulled
- Qdrant (Docker or local install)

---

## How to Run (3 Terminals)

### Terminal 1 — Qdrant Vector Database

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Terminal 2 — Backend (FastAPI)

```bash
cd backend

# First time only: create venv and install deps
python3 -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt
python3 -m spacy download en_core_web_sm

# Create .env file (first time only)
cat > .env << 'EOF'
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=lexved-index
GROQ_API_KEY=
COHERE_API_KEY=
EOF

# Run the server
source venv/bin/activate
python3 app.py
```

Backend starts at: **http://localhost:5000**

### Terminal 3 — Frontend (Next.js)

```bash
cd frontend

# First time only: install deps
npm install

# Run dev server
npm run dev
```

Frontend starts at: **http://localhost:3000**

### Terminal 4 (optional) — Ollama

```bash
# Pull Llama 3 if not already done
ollama pull llama3

# Ollama runs automatically as a service, but if needed:
ollama serve
```

---

## Quick Start (After Initial Setup)

Once everything is installed, you only need 3 commands in 3 terminals:

```
Terminal 1:  docker run -p 6333:6333 qdrant/qdrant
Terminal 2:  cd backend && source venv/bin/activate && python3 app.py
Terminal 3:  cd frontend && npm run dev
```

Then open **http://localhost:3000** in your browser.

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | /api/chat | Streaming legal Q&A (multi-turn) |
| POST | /api/ingest | Upload PDF for ingestion |
| GET | /api/files | List indexed documents |
| GET | /api/health | System health check |
| GET | /api/metrics | Fetch evaluation results |
| GET | /api/history | Query history log |
| DELETE | /api/history | Clear history |
| GET/POST | /api/settings/embedding_model | Get/Set embedding model |
| GET/POST | /api/settings/vector_db | Get/Set vector database |
| GET/POST | /api/settings/generation_model | Get/Set generation model |
| POST | /api/workflow/evaluate | Trigger 24-KPI benchmark |
| POST | /api/workflow/comparative | Benchmark all 6 models |
| GET | /api/comparative | Fetch comparative results |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+K | Focus search input |
| Ctrl+M | Toggle Metrics Dashboard |
| Ctrl+H | Toggle Research History |
| Escape | Close any open modal |

## Docker Compose (Alternative)

Run everything with a single command:

```bash
docker-compose up --build
```

This starts Qdrant, Backend, and Frontend automatically.

Note: Ollama must still be running separately on the host machine.

## Evaluation Metrics (M1-M24)

| ID | Metric | Method |
|----|--------|--------|
| M1 | Embedding Latency | Timer |
| M2 | Index Point Count | DB Stats |
| M3 | Retrieval Latency | Timer |
| M4 | Cosine Similarity | SentenceTransformers |
| M5 | Recall@K | Category Match |
| M6-M8 | ROUGE-1/2/L | rouge-score |
| M9 | METEOR | HuggingFace evaluate |
| M10 | BLEU | HuggingFace evaluate |
| M11 | Semantic Score | Cosine(GT, Ans) |
| M12 | BERTScore F1 | bert-score |
| M13 | Hallucination Rate | DeBERTa NLI |
| M14 | Faithfulness | DeBERTa NLI |
| M15 | Factual Consistency | LLM Judge |
| M16 | E2E Latency | Timer |
| M17 | Token Gen Latency | Timer/Tokens |
| M18 | Cost Per Query | Token Estimation |
| M19 | RAM Utilization | psutil |
| M20 | Citation Accuracy | LLM Judge |
| M21 | Term Precision | SpaCy NER |
| M22 | Precedent Match | LLM Judge |
| M23 | Regulatory Alignment | LLM Judge |
| M24 | Jurisdictional Compliance | LLM Judge |
