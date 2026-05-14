# LexVed: Detailed Project Report (DPR)

## 1. Project Objective
LexVed is an institutional-grade, multi-agent Retrieval-Augmented Generation (RAG) platform engineered specifically for complex legal document analysis. The primary objective of this project is to solve the critical challenges of hallucination, lack of context, and inefficient retrieval in standard Large Language Model (LLM) implementations within the legal domain. 

By orchestrating a highly specialized ingestion, retrieval, and generation pipeline, LexVed guarantees that all AI-generated legal insights are strictly bounded by verifiable jurisprudence, providing exact citations, reasoning transparency, and deterministic accuracy.

---

## 2. System Architecture Overview

The system operates across four primary layers:
1. **Ingestion Layer:** Processes raw legal PDFs, applies Named Entity Recognition (NER) for redaction and semantic tagging, and chunks data intelligently.
2. **Retrieval Layer:** A dual-database (Pinecone / Qdrant) hybrid retrieval engine combining Dense vector search with BM25 Sparse search, culminating in Reciprocal Rank Fusion (RRF) and CrossEncoder reranking.
3. **Multi-Agent Generation Layer:** Dynamically routes queries to optimal LLMs, generates explicit reasoning chains, and synthesizes final responses with strict anti-hallucination constraints.
4. **Presentation Layer:** A premium, glassmorphic Next.js interface providing real-time streaming, evaluation metric dashboards, and citation cards.

```mermaid
graph TD
    A[Legal PDFs] --> B[SpaCy NER Processing]
    B --> C[Semantic Chunking]
    C --> D[Embedding Model]
    D --> E[(Pinecone / Qdrant)]
    
    F[User Query] --> G{Dynamic Router Agent}
    G -->|Universal| H[Universal Mode Inference]
    G -->|Agentic| I[Reasoning Agent]
    
    I --> J[Hybrid Retrieval Engine]
    E --> J
    J --> K[RRF & CrossEncoder Rerank]
    K --> L[Synthesis Agent]
    
    H --> M[Final Response + Citations]
    L --> M
```

---

## 3. The Primitive Pipeline vs. The Enhanced Pipeline

To demonstrate the efficacy of the advanced features, LexVed maintains a standalone "Primitive Pipeline" for comparative benchmarking against the production "Enhanced Pipeline."

### 3.1 The Primitive Pipeline (Baseline)
The Primitive Pipeline represents a standard, out-of-the-box RAG implementation. 

**Workflow:**
1. A query is received.
2. The query is embedded using a single, static embedding model.
3. A simple K-Nearest Neighbors (KNN) search is performed against a local vector database.
4. The retrieved text is passed directly into a zero-shot prompt.
5. The LLM generates a response in a single pass.

```mermaid
graph LR
    Q[Query] --> E[Embed] --> S[Vector Search] --> P[Prompt Construction] --> LLM[Generation]
```

**Limitations of the Primitive Pipeline:**
* **Hallucinations:** Without explicit reasoning steps, the LLM often conflates its pre-trained knowledge with the retrieved context.
* **Retrieval Blindspots:** Relying solely on dense embeddings misses exact-keyword matches (e.g., specific statute numbers or docket IDs).
* **Latency Bottlenecks:** Routing all queries, regardless of complexity, to a massive LLM results in unnecessary computational overhead.

### 3.2 The Enhanced Pipeline (Production)
The Enhanced Pipeline addresses every limitation of the Primitive Pipeline through a multi-tiered, agentic approach.

```mermaid
graph TD
    subgraph "Phase 1: Dynamic Routing"
        Q[User Query] --> R[Router Agent]
        R -->|Classification| D{Universal or Agentic?}
    end
    
    subgraph "Phase 2: Advanced Retrieval"
        D --> H[Hybrid Search Engine]
        H --> DENSE[Dense Vector Search]
        H --> SPARSE[BM25 Sparse Search]
        DENSE --> FUSE[Reciprocal Rank Fusion]
        SPARSE --> FUSE
        FUSE --> RERANK[CrossEncoder Reranking]
    end
    
    subgraph "Phase 3: Multi-Agent Generation"
        RERANK --> REASON[Reasoning Agent]
        REASON --> |Logical Chain| SYNTHESIS[Synthesis Agent]
    end
    
    SYNTHESIS --> OUT[Final Output with Citations]
```

---

## 4. Key Features of the Enhanced Pipeline

### A. Multi-Agent Architecture
Instead of generating an answer in a single pass, the Enhanced Pipeline splits the cognitive load across three specialized agents:
1. **Routing Agent:** Instantly evaluates the complexity of the user's query.
2. **Reasoning Agent:** Mandated to read the retrieved context and "think aloud." It drafts a step-by-step logical chain, actively searching for contradictions, precedents, and facts. It is strictly constrained from utilizing outside knowledge.
3. **Synthesis Agent:** Acts as the Senior Legal Counsel. It takes the raw logical chain from the Reasoning Agent and drafts a cohesive, professional, and authoritative final response, embedding exact citations.

### B. Dynamic Model Orchestration
LexVed utilizes a sophisticated multi-model orchestration strategy to balance speed, reasoning depth, and operational efficiency:
* **Universal Mode:** Leverages `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, or `qwen-2.5-32b` for rapid, low-latency responses. This mode provides a hybrid knowledge experience, blending the institutional document repository with the model's internal legal expertise.
* **Agentic Mode:** Escalates to high-parameter models like `llama-3.3-70b-versatile`, `qwen2.5:70b`, or `llama3:70b` for multi-agent reasoning chains and deep, logical deduction.
* **Inference Versatility:** Supports local deployment via Ollama (Mistral, Phi-3, Qwen) and high-speed cloud inference via Groq.
* **Infrastructure Parity:** Supports cross-infrastructure benchmarking across six distinct embedding models (MPNet, MiniLM, DistilBERT, E5-Large, BGE-M3, and Cohere) on both Pinecone and Qdrant.

### C. Hybrid Retrieval and Reranking
Dense vector search excels at understanding semantic intent, while sparse search (BM25) excels at exact keyword matching. LexVed executes both simultaneously, merges the results using Reciprocal Rank Fusion (RRF), and then passes the fused list through a CrossEncoder to re-score the chunks based on deep contextual relevance, guaranteeing that the most accurate legal precedent is fed to the LLM.

### D. Hierarchical Sub-Indexing
During ingestion, documents are categorically tagged (e.g., Civil, Criminal, Constitutional). At query time, the system can selectively filter the vector database, preventing context contamination across distinct legal domains.

### E. Evaluation Fingerprinting and Caching
The platform features an automated 24-KPI benchmark suite. To preserve computational resources, LexVed fingerprints the dataset corpus. If an evaluation is triggered without any new PDFs having been ingested, the system bypasses the redundant LLM execution and instantly serves the cached benchmark results.

### G. Agentic Hardening & Context Continuity
In the latest version, LexVed has been hardened for institutional audit readiness with three key memory and persona features:
* **Intelligent Query Condensation:** A specialized "Condensation Engine" rephrases multi-turn follow-up questions into standalone, context-aware queries. This prevents the "Context Drift" typical in RAG systems when users ask vague questions like "tell me more."
* **Strict Topic Locking:** The Reasoning Agent is now strictly constrained to the subject matter established in the conversation history, preventing name-collision hallucinations (e.g., confusing two different cases with the same first name).
* **Persona Refinement:** The "Synthesis Counsel" has been stripped of generic AI placeholders and formal letter templates, providing a more natural, human-expert-like interaction.

---

## 5. Development Timeline & Session Logs

### Session: Institutional RAG Pipeline Hardening (May 14, 2026)
* **Vector DB Restoration:** Successfully re-connected the production "Hybrid Index" (19,483 vectors) after identifying it as the gold-standard repository.
* **UI/UX Premium Polish:** Implemented automatic scrolling for streaming and premium glassmorphic scrollbars for reasoning chains.
* **Authentication Sync:** Synchronized JWT display names to enable personalized AI greetings ("Counsel Arjun").
* **Ingestion Integrity:** Patched the PDF processing pipeline to extract and persist physical page numbers across both Pinecone and Qdrant.

