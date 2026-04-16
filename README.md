# LexVed: Institutional Legal Intelligence

LexVed is a next-generation Legal RAG (Retrieval-Augmented Generation) platform designed for institutional research, featuring hierarchical sub-indexing, parallel ingestion with SpaCy NER, and a premium glassmorphic UI.

---

## 1. Unified Multi-DB Architecture
LexVed now features a unified infrastructure layer that allows dynamic switching between vector database providers directly from the browser:
*   **Dynamic Routing:** Seamlessly transition between **Pinecone** (Cloud-Native) and **Qdrant** (Local/On-Prem) without restarting the system.
*   **Embedding Omnitrix:** A 3D model selection dial supporting MPNet, MiniLM L6, and DistilBERT with automatic dimension synchronization (384 vs 768).

## 2. Institutional Evaluation Suite
The platform includes an automated auditing engine (`run_metrics.py`) that executes parallel ingestion and benchmarking:
*   **Sample Ingestion:** 10-PDF ground truth verification cycle.
*   **Full Spectrum Audit:** 24 KPIs across Retrieval, Quality, AI-Judge, and Efficiency.
*   **PDF Exports:** High-fidelity institutional reports generated for each model configuration (DistilBERT, MiniLM L6, MPNet).

## 3. Mission-Critical Evaluation (M1-M24)
LexVed incorporates an automated auditing suite (`run_metrics.py`) that evaluates the RAG pipeline across 24 KPIs:
*   **Retrieval:** Latency, Point Count, Similarity, Recall@K (M1-M5).
*   **Quality:** ROUGE, BLEU Score, METEOR, BERTScore (M6-M12).
*   **AI-Judge:** Factual Consistency, Faithfulness, Hallucination Rate (M13-M15).
*   **Efficiency:** Throughput, CPU/RAM Delta (M16-M19).
*   **Legal Precision:** Citation Accuracy, Terminology, Precedent Match (M20-M24).

## 4. Deployment
LexVed is optimized for institutional deployment:
*   **Frontend:** Next.js with Framer Motion and Three.js elements.
*   **Backend:** FastAPI with process-level monitoring (PID-based state tracking).
*   **Infrastructure:** Unified `main` branch supporting full infrastructure flexibility.

---
**LexVed | Institutional Legal Intelligence**
**Technical Project Report — Version 3.0 — April 2026**
