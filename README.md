# LexVed: Advanced Legal Intelligence Platform

A production-grade Retrieval-Augmented Generation (RAG) system engineered for institutional legal document analysis. LexVed features a multi-agent cognitive architecture, dynamic model routing, a dual-database hybrid retrieval pipeline, and an integrated 24-KPI automated benchmark suite.

## Core Features and Capabilities

### 1. Multi-Agent Architecture
LexVed abandons the traditional single-pass LLM generation in favor of a specialized, multi-agent cognitive workflow:
* **Routing Agent:** Evaluates query complexity in real-time, routing simple tasks to lightning-fast models and complex analytical tasks to heavy-weight reasoning models.
* **Reasoning Agent:** Analyzes the retrieved legal context and drafts a step-by-step logical deduction chain. It is strictly constrained by anti-hallucination prompts to prevent the fabrication of case law.
* **Synthesis Agent:** Acts as the Senior Legal Counsel, transforming the raw logical chain into a highly cohesive, authoritative, and properly cited final legal opinion.

### 2. Intelligent Context Memory
LexVed maintains conversation continuity across multi-turn interactions:
* **Query Condensation Engine:** Automatically rephrases follow-up questions into standalone queries to maintain retrieval accuracy.
* **Contextual Persistence:** Pass-through memory ensures the reasoning and synthesis agents stay locked onto the case subject established in previous messages.

### 3. Data Integrity & Fingerprinting
LexVed ensures 100% data integrity for large-scale legal corpuses:
* **SHA-256 Fingerprinting:** Every PDF is cryptographicly fingerprinted before ingestion. The system automatically identifies and skips duplicate files, preventing vector database bloat and redundant processing.
* **Content-Aware Caching:** Ingestion caches are indexed by content hash rather than filename, ensuring modifications to documents are detected even if filenames remain unchanged.

### 4. Advanced Hybrid Retrieval
The platform utilizes a multi-tiered retrieval strategy to guarantee maximum recall and precision:
* **Production Hybrid Index:** Active connection to the 19,483-vector production repository (MPNet-based).
* **Dense Vector Search:** Understands the semantic intent of complex legal queries using high-dimensional embeddings (Pinecone/Qdrant).
* **Sparse Search (BM25):** Ensures critical exact-keyword matches, such as specific statute sections or docket numbers, are never missed.
* **Reciprocal Rank Fusion (RRF) & CrossEncoder Reranking:** Merges dense and sparse results and re-scores the combined list based on deep contextual relevance before feeding the context to the LLMs.

* **Universal Mode:** Utilizes `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, or `qwen-2.5-32b` for rapid, low-latency inference. This mode is not restricted to ingested documents and can draw from the model's vast pre-trained legal knowledge when needed.
* **Agentic Mode:** Escalates to high-parameter models like `llama-3.3-70b-versatile`, `qwen2.5:70b`, or `llama3:70b` for deep, multi-faceted analysis requiring extensive logical deduction. Strictly constrained to provided context.
* **Evaluation Node:** Employs `llama-3.1-8b-instant` on Groq for high-performance KPI judging.

### 5. Enterprise Interface
The frontend is built on Next.js 14 and Framer Motion, delivering a premium, glassmorphic UI ("All Black But Gold" aesthetic). It features:
* Real-time streaming of Agentic Reasoning Chains.
* **Smart Citations:** Clicking a citation now opens the source PDF directly in a new tab, automatically jumping to the cited page using secure one-time URL tokens.
* **GPU-Aware Loading:** Transparent UI feedback during local model cold-starts ("Loading neural weights into local GPU memory...") eliminates perceived latency frustration.
* A comprehensive Metrics Dashboard for benchmark visualization.

## Technical Stack

| Layer | Technology |
|-------|-----------|
| **Language Models** | Llama 3.1 8B, Llama 3.3 70B, Mixtral 8x7B, Qwen 2.5 (70B/32B), Llama 3 70B, Mistral, Phi-3 |
| **Embeddings** | MPNet, MiniLM, DistilBERT, E5-Large-Instruct, BGE-M3, Cohere Embed v3 |
| **Vector Databases** | Qdrant (Self-Hosted), Pinecone (Serverless) |
| **Retrieval Engine** | BM25, Reciprocal Rank Fusion, CrossEncoder |
| **Backend Framework** | FastAPI (Python 3.10) |
| **Frontend Framework** | Next.js 14, React, TailwindCSS, Framer Motion |
| **Evaluation Suite** | HuggingFace evaluate, DeBERTa NLI, SpaCy NER |

## System Prerequisites

- Python 3.10+
- Node.js 18+
- Docker (for Qdrant deployment)
- API Keys: Pinecone, Groq, Cohere

## Initialization and Setup

### 1. Vector Database Deployment
Start the local Qdrant instance:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Backend Environment
Navigate to the backend directory, initialize the environment, and start the server:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt
python3 -m spacy download en_core_web_sm

# Environment Configuration (.env required)
# PINECONE_API_KEY=your_key
# PINECONE_INDEX_NAME=lexved-index
# GROQ_API_KEY=your_key
# COHERE_API_KEY=your_key

python3 app.py
```
The backend initializes at `http://localhost:5000`.

### 3. Frontend Environment
Navigate to the frontend directory, install dependencies, and launch the application:
```bash
cd frontend
npm install
npm run dev
```
The frontend initializes at `http://localhost:3000`.

## Interactive Operations

### Fast vs. Agentic Mode
The interface includes a toggle switch allowing users to dictate the execution pipeline:
* **Fast Mode (Disabled):** Triggers the legacy, single-pass generation pipeline for immediate answers.
* **Agentic Mode (Enabled):** Activates the Multi-Agent architecture, generating visible reasoning chains and deep contextual synthesis.

### Pipeline Comparisons
The Metrics Dashboard includes a dedicated "Pipelines" tab, allowing administrators to execute and compare the performance of the Baseline Primitive Pipeline against the Enhanced Multi-Agent Pipeline across a 24-KPI benchmark suite.

## Institutional Audit Metrics (M1-M24)

The platform evaluates system integrity across 24 distinct dimensions, including:
* **M1 - M3:** Infrastructure Latency (Embedding, Indexing, Retrieval).
* **M4 - M5:** Retrieval Quality (Cosine Similarity, Recall@K).
* **M6 - M12:** Lexical and Semantic Precision (ROUGE, BLEU, METEOR, BERTScore).
* **M20 - M24:** Legal Verification (Citation Accuracy, Precedent Match, Regulatory Alignment).

## Embedding Models (6)

| Model | Dimensions | Source |
|-------|-----------|--------|
| multi-qa-mpnet-base-cos-v1 | 768 | SentenceTransformers |
| multi-qa-MiniLM-L6-cos-v1 | 384 | SentenceTransformers |
| multi-qa-distilbert-cos-v1 | 768 | SentenceTransformers |
| BAAI/bge-m3 | 1024 | HuggingFace |
| intfloat/multilingual-e5-large-instruct | 1024 | HuggingFace |
| Cohere embed-english-v3.0 | 1024 | Cohere API |

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
