# LexVed: Detailed Project Report (DPR)

**Technical Specification — Version 2.0 — April 2026**

---

## Abstract
This report presents the design, architecture, and development rationale of **LexVed**, an AI-driven private legal intelligence system. LexVed delivers high-precision legal research by combining a Next.js 14 frontend, FastAPI microservices, Qdrant vector repository, and a local-only Llama 3 8B inference engine.

The system addresses the critical problem of data privacy in legal technology by ensuring that no sensitive case data ever leaves the local machine. The architecture follows a modular pattern, enabling independent scaling of the ingestion engine, retrieval core, and AI reasoning layer.

---

## 1. System Objectives
*   **Privacy-First Legal AI:** Eliminate cloud API dependencies to ensure absolute document confidentiality.
*   **Hierarchical Precision:** Implement domain-isolated sub-indexing for Criminal and Civil law.
*   **Institutional Aesthetics:** Deliver a "Premium Intelligence Cockpit" UI for corporate legal environments.
*   **Systemic Accountability:** Integrate a 24-metric evaluation framework (M1-M24) for automated performance auditing.

---

## 2. System Architecture

### 2.1 High-Level Architecture
LexVed follows a layered microservices pattern:

```text
Browser Client (Intelligence Cockpit)
         |
Next.js Frontend (Framer Motion + Tailwind)
         |
FastAPI Gateway (Intelligence Core)
         |
 ----------------------------------------
 | Ingestion Engine | Retrieval Core    |
 | PII Redactor     | LLM Reasoning Node|
 ----------------------------------------
         |
Qdrant Vector DB
```

### 2.2 Technology Stack
| Layer | Technology |
| :--- | :--- |
| **Frontend** | Next.js 14 (React) |
| **Styling** | Tailwind CSS (Institutional Gold/Black) |
| **Backend API** | FastAPI (Python 3.10) |
| **Vector DB** | Qdrant (Dockerized) |
| **LLM Node** | Llama 3 8B (via Ollama) |
| **Embeddings** | all-mpnet-base-v2 |
| **NER Engine** | spaCy (en_core_web_sm) |

---

## 3. Module Descriptions

### 3.1 Ingestion Engine
Handles the full lifecycle of legal document processing:
*   **Parallel Extraction:** High-speed text extraction from large PDF volumes.
*   **Automatic Category Detection:** Sorts documents into Criminal/Civil sub-indices.
*   **PII Redaction:** Mandatory scrubbing of sensitive data (Aadhaar, PAN, Names) before storage.

### 3.2 Retrieval Core
The retrieval core uses **Hybrid Semantic Search**:
*   **Domain Isolation:** Filters noise by restricting search to relevant legal categories.
*   **Context Assembly:** Ranks and formats top-k legal chunks with exact page-level citations.

### 3.3 LLM Reasoning Node
A specialized prompt-engineered layer for Llama 3:
*   **Citation Enforcement:** Strict rules ensuring the LLM only answers based on provided context.
*   **Legal Tone:** Calibrated for formal, institutional communication.

---

## 4. Mission-Critical Metrics (M1-M24)
LexVed includes an automated auditing suite (`run_metrics.py`) that evaluates the RAG pipeline against 24 key performance indicators:

| ID | Category | Metric | Goal |
| :--- | :--- | :--- | :--- |
| **M1-M3** | Efficiency | Latencies | Minimize time-to-first-token |
| **M4** | Retrieval | Cosine Similarity | > 0.7 Avg Semantic Match |
| **M6-M12** | Quality | ROUGE / BLEU / BERTScore | Maximize semantic alignment |
| **M14** | Accuracy | Faithfulness | 100% (Zero Hallucination) |
| **M20** | Legal | Citation Accuracy | Correct page mapping |
| **M21** | Legal | Terminology Precision | Accurate use of legal jargon |

---

## 5. Security & Data Sovereignty
*   **Local Inference:** All LLM weights remain on-premise. No OpenAI/Anthropic cloud leaks.
*   **Encrypted Storage:** All embeddings and metadata are stored in a hardened Qdrant container.
*   **PII Sanitization:** The spaCy NLP layer ensures zero personal identifying information is stored in the vector space.

---

## 6. Roadmap
### Near-term
*   Integration of Multi-jurisdictional sub-indexing.
*   Real-time citation verification against official government gazettes.

### Long-term
*   Computer Vision for analyzing hand-annotated legal evidence.
*   Decentralized private legal knowledge graphs.

## Institutional Performance Audit (M1-M24)
To ensure mission-critical precision, LexVed incorporates a 24-metric evaluation framework. This suit audits the RAG pipeline across retrieval latency, semantic similarity (BERTScore), and legal citation accuracy using an LLM-as-a-Judge protocol.

### Metric Overview
*   **Retrieval Performance (M1-M5):** Tracks embedding latency and vector counts.
*   **Answer Quality (M6-M15):** Measures ROUGE, BLEU, and Faithfulness.
*   **Legal-Specific Accuracy (M20-M22):** Validates legal terminology and case precedents.

### Professional PDF Export
The "LexVed Intelligence Cockpit" now features a professional export engine. Legal users can generate an institutional-grade PDF report of the latest performance audit with a single click, ensuring transparency and archive-readiness for legal audits.

---
**LexVed | Institutional Legal Intelligence**

---
**LexVed | Technical Project Report — Version 2.0 — April 2026**
