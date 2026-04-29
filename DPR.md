# LexVed: Dense Passage Retrieval (DPR) Framework

This document outlines the architecture for dense vector search, resilient evaluation, and multi-model benchmarking within the unified LexVed platform.

---

## 1. Unified Embedding Architecture
LexVed employs a dynamic embedding layer capable of synchronizing vector dimensions across 6 distinct model architectures, seamlessly selectable via the hardware-accelerated frontend carousel:
*   **MPNet Base:** High-precision Legal RAG baseline (768 Dimensions • Dense).
*   **MiniLM L6:** Efficient, low-latency retrieval (384 Dimensions • Lite).
*   **DistilBERT:** Balanced performance for complex semantic mapping (768 Dimensions • Balanced).
*   **BGE-M3:** State-of-the-art multilingual support (1024 Dimensions • Multilingual).
*   **E5-Mistral:** Instruction-tuned embeddings (4096 Dimensions • Instruct).
*   **Cohere v3.0:** Cloud-based enterprise embeddings (1024 Dimensions • API).

The system automatically manages collection state in both **Pinecone** and **Qdrant**, ensuring index parameters match the active model's dimensionality.

## 2. Ingestion & Inference Pipeline
*   **Parallel Processing:** SpaCy NER-driven extraction with hard-locked `ThreadPoolExecutor` optimization (max 2 workers) to prevent memory thrashing on local hardware.
*   **Resilient Generation:** The generation stream (`Ollama`) is wrapped with fallback heuristics. In the event of inference timeouts or connection drops during heavy multi-model benchmarks, the pipeline gracefully substitutes safe fallback responses instead of crashing the thread.
*   **Hierarchical Sub-indexing:** Metadata-rich points tagged by `category`, `subcategory`, and `page_number` for precise faceted search.
*   **Adaptive Uploaders:** Dynamic routing to `upsert` points into the active vector provider (Pinecone/Qdrant).

## 3. Institutional Audit (M1-M24)
The DPR evaluation suite provides a localized benchmarking tool (`run_metrics.py`) that uses a 10-PDF ground truth sample to calculate:
1.  **Semantic Precision:** Faithfulness and Factual Consistency via local Llama 3 LLM Judge.
2.  **Infrastructure KPIs:** Retrieval/Embedding Latency, E2E Latency, and Memory overhead.
3.  **Legal Accuracy:** Terminology alignment and citation integrity.

---
**Technical Specification — DPR Engine v3.1**
