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

## 2. Infrastructure & Modules

### 2.1 Mission-Critical Evaluation (M1-M24)
LexVed incorporates an automated auditing suite (`run_metrics.py`) that evaluates the RAG pipeline across 24 KPIs:
*   **Retrieval:** Latency, Point Count, Similarity, Recall@K (M1-M5).
*   **Quality:** ROUGE, BLEU Score, METEOR, BERTScore (M6-M12).
*   **AI-Judge:** Factual Consistency, Faithfulness, Hallucination Rate (M13-M15).
*   **Efficiency:** Throughput, CPU/RAM Delta (M16-M19).
*   **Legal Precision:** Citation Accuracy, Terminology, Precedent Match (M20-M24).

### 2.2 Institutional Export Engine
A robust PDF and CSV/Excel generation suite:
*   **PDF:** High-fidelity audit report with date-stamping and AES-256 storage metadata.
*   **CSV:** Raw audit data for spreadsheet integration and external compliance reviews.

### 2.3 Functional Sidebars
*   **Case Files:** Direct access to the local legal corpus and knowledge base files.
*   **Research History:** Log of verified and flagged institutional queries.

---

## 3. Deployment & Branch Strategy
LexVed follows a robust branch architecture for enterprise adaptability:
*   **main:** Production-ready baseline with Qdrant integration.
*   **qdrant:** Focused repository for Qdrant-optimized retrieval.
*   **pinecone:** Managed vector storage implementation for cloud-hybrid scenarios.

---
**LexVed | Institutional Legal Intelligence**
**Technical Project Report — Version 2.5 — April 2026**l 2026**
