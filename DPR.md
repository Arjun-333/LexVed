# LexVed: Detailed Project Report (DPR)

**Technical Specification — Version 2.5 — April 2026**

---

## Abstract
This report presents the design, architecture, and development rationale of **LexVed**, an AI-driven private legal intelligence system. LexVed delivers high-precision legal research by combining a Next.js frontend, FastAPI microservices, and a dual-repository vector strategy (Qdrant/Pinecone).

---

## 1. System Objectives
*   **Privacy-First Legal AI:** Eliminate cloud API dependencies for inference.
*   **Institutional Auditing:** Integrate a full 24-metric evaluation framework (M1-M24).
*   **Intelligence Cockpit:** Functional sidebars for Case Files and Research History.
*   **Hybrid Vector Strategy:** Branch-level isolation for Qdrant and Pinecone implementations.

---

## 2. Infrastructure & Modules: The 10-Layer Pipeline

LexVed operates through a specialized intelligence pipeline, ensuring 100% data sovereignty and page-level citation accuracy.

### 2.1 Data Management & Page Indexing
- **Layer 1: Legal Corpus:** Raw repository management for PDF, DOCX, and TXT files.
- **Layer 2: Ingestion Engine:** Parallel text extraction with automated domain isolation (Civil/Criminal).
- **Layer 3: PII Redactor:** spaCy NER-based scrubbing of sensitive institutional data (Aadhaar, PAN, etc.)

### 2.2 Semantic Foundation
- **Layer 4: MPNet Embedding:** Transformation of legal text into 768-dimensional semantic vectors using `all-mpnet-base-v2`.
- **Layer 5: Vector Repository:** Qdrant/Pinecone hierarchical sub-indexing with page-aware metadata storage.

### 2.3 Modular Intelligence Cockpit
- **Layer 6: UI Layer:** Premium Next.js 16 interface with real-time state management.
- **Layer 7: Auditing Layer (M1-M24):** Automated benchmarking engine tracking 24 mission-critical KPIs.

### 2.4 Inference & Output
- **Layer 8: Reasoning Node:** Local Llama 3 8B inference with strict factual grounding rules.
- **Layer 9: Streaming Engine:** Real-time NDJSON delivery with dynamic citation injection.
- **Layer 10: Institutional Export:** Professional PDF/CSV reporting with encrypted date-stamping.

---

## 3. Mission-Critical Evaluation (M1-M24)
LexVed incorporates an automated auditing suite (`run_metrics.py`) that evaluates the RAG pipeline across 24 KPIs:
*   **Retrieval:** Latency, Point Count, Similarity, Recall@K (M1-M5).
*   **Quality:** ROUGE, BLEU Score, METEOR, BERTScore (M6-M12).
*   **AI-Judge:** Factual Consistency, Faithfulness, Hallucination Rate (M13-M15).
*   **Efficiency:** Throughput, CPU/RAM Delta (M16-M19).
*   **Legal Precision:** Citation Accuracy, Terminology, Precedent Match (M20-M24).

---

## 3. Deployment & Branch Strategy
LexVed follows a robust branch architecture for enterprise adaptability:
*   **main:** Production-ready baseline with Qdrant integration.
*   **qdrant:** Focused repository for Qdrant-optimized retrieval.
*   **pinecone:** Managed vector storage implementation for cloud-hybrid scenarios.

---
**LexVed | Institutional Legal Intelligence**
**Technical Project Report — Version 2.5 — April 2026**l 2026**
