# LexVed: Dense Passage Retrieval (DPR) Framework

This document outlines the architecture for dense vector search and multi-model benchmarking within the unified LexVed platform.

---

## 1. Unified Embedding Architecture
LexVed employs a dynamic embedding layer capable of synchronizing vector dimensions across multiple model architectures:
*   **MPNet Base:** High-precision Legal RAG baseline (768 Dimensions).
*   **MiniLM L6:** Efficient, low-latency retrieval (384 Dimensions).
*   **DistilBERT:** Balanced performance for complex semantic mapping (768 Dimensions).

The system automatically manages collection state in both **Pinecone** and **Qdrant**, ensuring index parameters match the active model's dimensionality.

## 2. Ingestion Pipeline
*   **Parallel Processing:** SpaCy NER-driven extraction with ThreadPoolExecutor optimization.
*   **Hierarchical Sub-indexing:** Metadata-rich points tagged by `category`, `subcategory`, and `page_number` for precise faceted search.
*   **Adaptive Uploaders:** Dynamic routing to `upsert` points into the active vector provider (Pinecone/Qdrant).

## 3. Institutional Audit (M1-M24)
The DPR evaluation suite provides a localized benchmarking tool (`run_metrics.py`) that uses a 10-PDF ground truth sample to calculate:
1.  **Semantic Precision:** Faithfulness and Factual Consistency via Ollama.
2.  **Infrastructure KPIs:** Retrieval/Embedding Latency and Memory overhead.
3.  **Legal Accuracy:** Terminology alignment and citation integrity.

---
**Technical Specification — DPR Engine v3.0**
