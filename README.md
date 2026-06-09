# LexVed: Advanced Legal Intelligence Platform

A production-grade Retrieval-Augmented Generation (RAG) system engineered for institutional legal document analysis. LexVed features a multi-agent cognitive architecture, dynamic model routing, a dual-database hybrid retrieval pipeline, and an integrated 27-KPI automated benchmark suite.

## Core Features and Capabilities

### 1. Stateful Tool-Based Agent Architecture (LangGraph)
LexVed has transitioned from a rigid, multi-agent pipeline to a state-managed orchestrator built on **LangGraph**:
* **Autonomous Tool Use:** Instead of a hardcoded retrieve-then-synthesize flow, the system is guided by a stateful agent loop. The LLM evaluates the query and chooses dynamically from a suite of legal tools.
* **Dynamic Tool Suite:** Registered tools include:
  - `retrieve_documents`: Triggers the entire advanced hybrid retrieval engine (Dense + BM25 Sparse + RRF + CrossEncoder).
  - `extract_citations`: Regex and NER patterns to extract specific legal provisions (e.g. IPC, CrPC, AIR, SCC citations).
  - `extract_entities`: SpaCy-powered Named Entity Recognition to categorize parties, courts, dates, and locations.
  - `deidentify_text`: Automatic PII redact engine.
* **Stateful Message Memory:** Utilizes a graph state with an `add_messages` reducer to accumulate query inputs, agent thoughts, and tool execution outputs across sequential reasoning steps.

### 2. Intelligent Context Memory
LexVed maintains conversation continuity across multi-turn interactions:
* **Query Condensation Engine:** Automatically rephrases follow-up questions into standalone queries to maintain retrieval accuracy.
* **Contextual Persistence:** Pass-through memory ensures the agent stays locked onto the case subject established in previous messages.

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

### 5. Configurable Intelligence Engine
The system supports two core operational modes dynamically configured from backend metadata:
* **Standard Mode (Universal):** A fixed, lightning-fast RAG pipeline (retrieve → generate) leveraging `llama-3.1-8b-instant` or `mixtral-8x7b-32768`.
* **Agent Mode (Agentic):** Escalates to `llama-3.3-70b-versatile` inside the LangGraph state machine, allowing the AI to autonomously decide when to use tools, analyze data, and synthesize case law.
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
The Metrics Dashboard includes a dedicated "Pipelines" tab, allowing administrators to execute and compare the performance of the Baseline Primitive Pipeline against the Enhanced Multi-Agent Pipeline across a 26-KPI benchmark suite.

## Institutional Audit Metrics (M1-M31)

The platform evaluates system integrity across 31 distinct dimensions, including:
* **M1 - M3:** Infrastructure Latency (Embedding, Indexing, Retrieval).
* **M4 - M5:** Retrieval Quality (Cosine Similarity, Recall@5).
* **M6 - M12:** Lexical and Semantic Precision (ROUGE, BLEU, METEOR, BERTScore).
* **M13 - M15:** Grounding & Faithfulness (Factual Consistency Deviation, Faithfulness, Porter-Stemmed GT Coverage).
* **M20 - M24:** Legal Verification (Regex Citation Accuracy, Precedent Match, Regulatory Alignment).
* **M25 - M27:** Generation Efficiency (Prefill Latency, Time To First Token / TTFT, Generation Throughput).
* **M28 - M31:** Standard IR Benchmarks (Recall@10, MRR, nDCG@10, Precision@5).

## Embedding Models (4)

| Model | Dimensions | Source |
|-------|-----------|--------|
| multi-qa-MiniLM-L6-cos-v1 | 384 | SentenceTransformers |
| multi-qa-mpnet-base-cos-v1 | 768 | SentenceTransformers |
| multi-qa-distilbert-cos-v1 | 768 | SentenceTransformers |
| BAAI/bge-m3 | 1024 | HuggingFace |

## Evaluation Metrics (M1-M31)

| ID | Metric | Method |
|----|--------|--------|
| M1 | Embedding Latency | Timer |
| M2 | Index Point Count | DB Stats |
| M3 | Retrieval Latency | Timer |
| M4 | Cosine Similarity | SentenceTransformers |
| M5 | Recall@5 | Known Gold Chunk |
| M6-M8 | ROUGE-1/2/L | rouge-score |
| M9 | METEOR | HuggingFace evaluate |
| M10 | BLEU | HuggingFace evaluate |
| M11 | Semantic Score | Cosine(GT, Ans) |
| M12 | BERTScore F1 | bert-score |
| M13 | Factual Consistency Deviation | 1.0 - Faithfulness |
| M14 | Faithfulness | LLM Judge |
| M15 | GT Coverage (%) | Porter Stemmer Match |
| M16 | E2E Latency | Timer |
| M17 | Token Gen Latency | Timer/Tokens |
| M18 | Cost Per Query | Token Estimation |
| M19 | RAM Utilization | psutil |
| M20 | Citation Accuracy | Regex Citation Matcher |
| M21 | Term Precision | SpaCy NER |
| M22 | Precedent Match | LLM Judge |
| M23 | Regulatory Alignment | LLM Judge |
| M24 | Jurisdictional Compliance | LLM Judge |
| M25 | Prefill Latency | Groq Usage API |
| M26 | Time to First Token (TTFT) | Groq Stream Timer |
| M27 | Generation Throughput (Tokens/sec) | Timer / Token Count |
| M28 | Recall@10 | Known Gold Chunk |
| M29 | Mean Reciprocal Rank (MRR) | Reciprocal Rank |
| M30 | nDCG@10 | Normalized Discounted Cumulative Gain |
| M31 | Precision@5 | Known Gold Chunk |

## Swarm Mode & GPU Benchmarking

### 1. Collaborative Swarm Mode (LangGraph Swarm)
In the chat interface, queries starting with `/swarm ` bypass the single-agent pipeline and route directly to a collaborative multi-agent swarm:
* **Researcher Agent:** Dense vector & sparse keyword retrieval.
* **Extractor Agent:** Citation parsing and SpaCy entity tagging.
* **Drafting Counsel Agent:** Document synthesis.
* **Compliance Auditor Agent:** Critiques the generated draft against raw documents and requests revision loop-backs if any hallucination is detected.
* **PII Redactor Agent:** Redacts personal details before output.

### 2. A100 GPU Comparative Benchmark
A standalone script `backend/gpu_comparative_benchmark.py` is included for execution on A100 GPUs or remote Jupyter instances:
* Utilizes local **PyTorch/CUDA** matrix multiplication for local vector search.
* Queries Pinecone index directly with automatic index-creation failovers.
* Generates a side-by-side Markdown result table and a landscape PDF report `LexVed_GPU_Institutional_Audit.pdf`.

