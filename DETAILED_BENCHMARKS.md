# LexVed RAG System: Detailed Query-by-Query Benchmarking

LexVed features two offline benchmark scripts to perform side-by-side performance audits of the **Baseline Primitive** vs. **Enhanced Agentic** RAG pipelines:
1. `backend/pinecone_comparative_benchmark.py` (Hugging Face Llama-3.1-8B)
2. `backend/qwen_comparative_benchmark.py` (Local Ollama Qwen 2.5 7b)

We have updated both scripts to retain query-level execution traces and write streamlined comparison reports.

---

## 1. What is Captured for Each Query?
The scripts now capture and record the following information for every query:
* **The original legal query**
* **The ground truth answer (gold standard)**
* **The retrieved text context** (top-3 chunks parsed by each pipeline)
* **The generated answers** for both Primitive and Enhanced pipelines
* **The detailed KPI scores** for the query
* **The LLM Judge audit statements and supported evaluations** (fact-checking verification)

---

## 2. File Artifacts Generated

Running the benchmark scripts generates the following detailed artifacts:

### JSON Data Artifacts
* **`backend/gpu_comparative_results.json`** — Stores the model configurations, global metric summaries, and a full query-by-query breakdown under `primitive_details` and `enhanced_details`.
* **`backend/qwen_comparative_results.json`** — Stores the equivalent data for the Qwen 2.5 7b pipeline.

### Markdown Comparison Reports
* **`backend/pinecone_benchmark_detailed_report.md`** — A beautiful, side-by-side comparative report presenting every query, the ground truth, the actual generated responses, and the LLM Judge factual validation statement-by-statement.
* **`backend/qwen_benchmark_detailed_report.md`** — The equivalent side-by-side report generated for the local Qwen 2.5 7b model.

### PDF Report Cards
* **`backend/LexVed_Pinecone_Comparative_Audit.pdf`** — Landscape-formatted printable PDF listing the 31-KPI aggregate comparisons.
* **`backend/LexVed_Qwen_Comparative_Audit.pdf`** — Landscape-formatted printable PDF for Qwen 2.5 7b.

---

## 3. How to Run the Benchmarks

To execute the comparative benchmarks and generate these detailed files, navigate to the `backend` directory and activate the virtual environment:

```bash
cd backend
source venv/bin/activate
```

### Running the Pinecone Benchmark (Hugging Face LLM)
```bash
python3 pinecone_comparative_benchmark.py
```
*You will be prompted for API Keys/Tokens if they are not already set in your `.env` file, and you can choose the target embedding model.*

### Running the Qwen Benchmark (Ollama Local LLM)
Make sure Ollama is running locally with the Qwen 2.5 7b model pulled:
```bash
ollama pull qwen2.5:7b
python3 qwen_comparative_benchmark.py
```

---

## 4. Reading the Detailed Reports

The generated Markdown reports (`gpu_benchmark_detailed_report.md` & `qwen_benchmark_detailed_report.md`) follow this layout:

```markdown
## Query [N]: [Original Legal Question]

### Ground Truth Answer
> [Grounded Legal Rule / Standard]

### Side-by-Side Generated Answers
| Pipeline | Generated Answer | Key Metrics |
| --- | --- | --- |
| **Primitive** | [Raw search list style response] | Faithfulness, Citation Acc, Latency |
| **Enhanced** | [Authoritative counsel style response] | Faithfulness, Citation Acc, Latency |

### Retrieved Context Comparison
#### Primitive Retrieved Chunks (Top 3)
1. *[Text chunk]*
2. ...

#### Enhanced Retrieved Chunks (Top 3)
1. *[Text chunk]*
2. ...

### LLM Judge Audit Verification
**Primitive Pipeline Statements Check:**
- "Statement 1" (✅ Supported / ❌ Not Supported)
- ...

**Enhanced Pipeline Statements Check:**
- "Statement 1" (✅ Supported / ❌ Not Supported)
- ...
```

This streamlined output provides total transparency over why the Enhanced pipeline achieves higher legal compliance, better citation precision, and superior factual consistency compared to the primitive pipeline.
