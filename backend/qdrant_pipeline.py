"""
LexVed Primitive Pipeline — Pinecone + Groq Architecture
Mirrors the original Colab design: per-model Pinecone indexes, Groq for generation.
M20-M24 scored by Groq LLM judge (same as Enhanced pipeline).
"""
import os, re, sys, time, json, csv, warnings
import numpy as np
import pandas as pd
import psutil
import requests
import tiktoken
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

import nltk
for r in ['wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
    try: nltk.data.find(f'corpora/{r}')
    except LookupError: nltk.download(r, quiet=True)

from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score_fn
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")
from transformers import logging as hfl; hfl.set_verbosity_error()

# ── Qdrant ──────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
qc = QdrantClient(url="http://localhost:6333")

# ── Config ────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent
BASE_PDF_DIR = BACKEND_DIR / "data" / "PDF"
CACHE_FILE   = BACKEND_DIR / "data" / "primitive_chunk_cache.json"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

MODEL_DIMENSIONS = {
    "multi-qa-mpnet-base-cos-v1":  768,
    "multi-qa-MiniLM-L6-cos-v1":  384,
    "multi-qa-distilbert-cos-v1": 768,
    "BGE-M3":                     1024,
}
MODEL_INDEX_MAP = {
    "multi-qa-mpnet-base-cos-v1":  "qdrant-mpnet",
    "multi-qa-MiniLM-L6-cos-v1":  "qdrant-minilm",
    "multi-qa-distilbert-cos-v1": "qdrant-distilbert",
    "BGE-M3":                     "qdrant-bge-m3",
}

enc = tiktoken.encoding_for_model("gpt-4o-mini")

STATUS_PATH = BACKEND_DIR / "primitive_evaluation_results.json"

def _write_status(d: dict):
    with open(STATUS_PATH, "w") as f:
        json.dump(d, f, indent=2)

# ── Qdrant index management ─────────────────────────────────────────
def get_or_create_index(name: str, dim: int):
    existing = [c.name for c in qc.get_collections().collections]
    if name not in existing:
        qc.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        print(f"[Qdrant] Collection '{name}' created (dim={dim})")
    else:
        print(f"[Qdrant] Collection '{name}' exists.")
    return name

# ── PDF chunking ──────────────────────────────────────────────────────
def extract_chunks(pdf_path: Path, chunk_size=200):
    try:
        import fitz
        doc  = fitz.open(str(pdf_path))
        text = "\n".join([p.get_text("text") for p in doc])
        doc.close()
    except Exception as e:
        print(f"[Primitive] Skipping {pdf_path.name}: {e}")
        return []

    text = re.sub(r"[*_]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    citation = re.compile(r"(Section\s\d+[A-Za-z]*|Sec\.\s\d+[A-Za-z]*|\d+\s?Cr\.?\s?\d+)", re.IGNORECASE)
    sentences = re.split(r'(?<=[.?!]) +|\n+', text)

    chunks, buf = [], ""
    for sent in sentences:
        for part in citation.split(sent):
            seg = part.strip()
            if not seg: continue
            if len(buf.split()) + len(seg.split()) < chunk_size:
                buf += " " + seg
            else:
                if buf.strip(): chunks.append({"text": buf.strip(), "source": str(pdf_path)})
                buf = seg
    if buf.strip(): chunks.append({"text": buf.strip(), "source": str(pdf_path)})
    return chunks

def load_all_chunks() -> list:
    """Load/cache chunks from all 516 PDFs."""
    cache = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f: cache = json.load(f)
            print(f"[Primitive] Cache: {len(cache)} PDFs already parsed.")
        except: pass

    pdf_files = list(BASE_PDF_DIR.rglob("*.pdf"))
    print(f"[Primitive] Found {len(pdf_files)} PDFs.")
    all_chunks, updated = [], False

    for p in tqdm(pdf_files, desc="Parsing PDFs"):
        k = str(p)
        if k in cache:
            all_chunks.extend(cache[k])
        else:
            try:
                from src.ingestion.pdf_processor import process_chunks_batch
                extracted = process_chunks_batch(extract_chunks(p))
            except Exception:
                extracted = extract_chunks(p)
            cache[k] = extracted
            all_chunks.extend(extracted)
            updated = True

    if updated:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f: json.dump(cache, f)
        print(f"[Primitive] Cache saved ({len(cache)} PDFs).")

    print(f"[Primitive] Total chunks: {len(all_chunks)}")
    return all_chunks, pdf_files

# ── Embed & Upsert to Qdrant ─────────────────────────────────────────
def embed_and_upsert(chunks, embedder, model_name, collection_name, batch_size=96):
    # Check if already populated
    existing = qc.get_collection(collection_name).points_count
    if existing >= len(chunks) * 0.9:
        print(f"[Qdrant] Collection already has {existing} vectors — skipping upsert.")
        emb_time = 0.0
        return emb_time, existing

    texts = [c["text"] for c in chunks]
    is_cohere = hasattr(embedder, "encode_query")
    print(f"[Primitive] Embedding {len(texts)} chunks...")
    t0 = time.time()
    vecs = embedder.encode(texts, show_progress_bar=True, batch_size=batch_size)
    emb_time = time.time() - t0
    print(f"[Primitive] Embedding done in {emb_time:.1f}s")

    batch = []
    for i, vec in enumerate(tqdm(vecs, desc="Upserting to Qdrant")):
        batch.append(
            PointStruct(
                id=i,
                vector=vec.tolist(),
                payload={"text": chunks[i]["text"], "source": chunks[i]["source"]}
            )
        )
        if len(batch) >= batch_size:
            qc.upsert(collection_name=collection_name, points=batch)
            batch = []
    if batch:
        qc.upsert(collection_name=collection_name, points=batch)
    print(f"[Qdrant] {len(vecs)} vectors upserted.")
    return emb_time, len(vecs)

# ── Groq generation ───────────────────────────────────────────────────
def generate_with_groq(prompt: str, model="llama-3.1-8b-instant") -> str:
    if not GROQ_API_KEY:
        return "[Error: No GROQ_API_KEY]"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    for attempt in range(3):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 2: time.sleep(2**attempt)
            else: return f"[Groq Error: {e}]"

# ── Groq LLM judge (M14, M15, M20-M24) ───────────────────────────────
def judge_with_groq(query, ground_truth, model_answer, context) -> dict:
    defaults = {"faithfulness": 50, "citation_acc": 50, "term_precision": 50,
                "precedent_match": 50, "factual_consistency": 50, "bias_score": 10,
                "regulatory_alignment": 50, "jurisdictional_comp": 50}
    prompt = f"""You are an expert legal auditor. Evaluate the RAG output on 8 metrics.
QUERY: {query}
GROUND TRUTH: {ground_truth}
MODEL ANSWER: {model_answer}
CONTEXT: {context[:5000]}

Return ONLY valid JSON with integer scores 0-100:
{{"faithfulness":75,"citation_acc":60,"term_precision":80,"precedent_match":50,"factual_consistency":70,"bias_score":5,"regulatory_alignment":85,"jurisdictional_comp":90}}"""
    if not GROQ_API_KEY: return defaults
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.1, "response_format": {"type": "json_object"}}
    try:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            raw = r.json()["choices"][0]["message"]["content"]
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                parsed = {k.lower(): v for k, v in json.loads(m.group(0)).items()}
                return {**defaults, **parsed}
    except Exception as e:
        print(f"[Primitive] Judge error: {e}")
    return defaults

def _jval(judge, key, default=50.0):
    try: return float(str(judge.get(key, default)).strip()) / 100.0
    except: return default / 100.0

# ── Full evaluation for one model ─────────────────────────────────────
def run_evaluation(model_name, embedder, index, chunks, queries, gts, emb_time, index_size):
    is_cohere = hasattr(embedder, "encode_query")
    preds, ret_texts_all, q_vecs = [], [], []
    r_times, g_times = [], []

    for q in tqdm(queries, desc=f"Evaluating {model_name}"):
        # Encode query
        if is_cohere: q_vec = embedder.encode_query([q])[0]
        else: q_vec = embedder.encode([q], show_progress_bar=False)[0]

        # Retrieve from Qdrant
        t0 = time.time()
        res = qc.search(
            collection_name=index,
            query_vector=q_vec.tolist(),
            limit=5
        )
        ret = [hit.payload.get("text", "") for hit in res]
        rt = time.time() - t0

        # Generate with Groq
        context_str = "\n\n".join(ret)
        prompt = (f"Answer the following query using ONLY the context below.\n\n"
                  f"Context:\n{context_str}\n\nQuery: {q}\nAnswer:")
        t1 = time.time()
        ans = generate_with_groq(prompt)
        gt_time = time.time() - t1

        preds.append(ans); ret_texts_all.append(ret)
        q_vecs.append(q_vec); r_times.append(rt); g_times.append(gt_time)

    # Compute metrics
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    df_list = []

    for i in range(len(preds)):
        r = rouge.score(gts[i], preds[i])
        try:
            _, _, F1 = bert_score_fn([preds[i]], [gts[i]], lang="en", verbose=False)
            bert_f1 = float(F1[0])
        except: bert_f1 = 0.0

        try: bleu = sentence_bleu([gts[i].split()], preds[i].split(), smoothing_function=smoothie)
        except: bleu = 0.0
        try: met = meteor_score([gts[i].split()], preds[i].split())
        except: met = 0.0

        if is_cohere: ctx_vecs = embedder.encode_query(ret_texts_all[i])
        else: ctx_vecs = embedder.encode(ret_texts_all[i], show_progress_bar=False) if ret_texts_all[i] else np.zeros((1, 1))
        cosine_sim = float(np.mean(cosine_similarity([q_vecs[i]], ctx_vecs))) if len(ctx_vecs) else 0.0
        sims_arr = cosine_similarity([q_vecs[i]], ctx_vecs)[0] if len(ctx_vecs) else []
        topk_acc = float(np.mean([1 if s > 0.8 else 0 for s in sims_arr])) if len(sims_arr) else 0.0

        ctx_joined = " ".join(ret_texts_all[i])
        try:
            _, _, F1c = bert_score_fn([preds[i]], [ctx_joined], lang="en", verbose=False)
            fcd = float(1 - F1c[0])
        except: fcd = 1.0

        mem = psutil.virtual_memory()
        context_tokens = len(enc.encode(ctx_joined))
        e2e = r_times[i] + g_times[i]

        # LLM Judge
        judge = judge_with_groq(queries[i], gts[i], preds[i], ctx_joined)

        df_list.append({
            "M1_Embedding_Time":              emb_time,
            "M2_Index_Size_Chunks":           index_size,
            "M3_Retrieval_Latency":           r_times[i],
            "M4_Cosine_Similarity":           cosine_sim,
            "M5_TopK_Accuracy":               topk_acc,
            "M6_ROUGE-1":                     r["rouge1"].fmeasure,
            "M7_ROUGE-2":                     r["rouge2"].fmeasure,
            "M8_ROUGE-L":                     r["rougeL"].fmeasure,
            "M9_Context_Length_Words":        len(ctx_joined.split()),
            "M9_Context_Tokens":              context_tokens,
            "M10_BLEU":                       bleu,
            "M11_METEOR":                     met,
            "M12_BERTScore_F1":               bert_f1,
            "M13_Factual_Consistency_Deviation": fcd,
            "M14_Faithfulness":               _jval(judge, "faithfulness"),
            "M15_Ground_Truth_Coverage":      _jval(judge, "factual_consistency") * 100,
            "M16_End_to_End_Latency":         e2e,
            "M17_Throughput_QPS":             round(1.0 / max(0.001, e2e), 4),
            "M18_CPU_Usage":                  psutil.cpu_percent(),
            "M19_RAM_Usage_GB":               round(mem.used / (1024**3), 2),
            "M20_Citation_Accuracy":          _jval(judge, "citation_acc"),
            "M21_Terminology_Precision":      _jval(judge, "term_precision"),
            "M22_Precedent_Coverage":         _jval(judge, "precedent_match") * 100,
            "M23_FCD_Score":                  _jval(judge, "regulatory_alignment"),
            "M24_Bias_Score":                 _jval(judge, "bias_score"),
            "GT_Reference":                   gts[i],
            "AI_Response":                    preds[i],
        })

    return pd.DataFrame(df_list)

# ── Per-model Pinecone entry point ────────────────────────────────────
CHOICE_MAP = {
    "1": ("multi-qa-mpnet-base-cos-v1",  "sentence-transformers/multi-qa-mpnet-base-cos-v1"),
    "2": ("multi-qa-MiniLM-L6-cos-v1",  "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
    "4": ("multi-qa-distilbert-cos-v1", "sentence-transformers/multi-qa-distilbert-cos-v1"),
    "5": ("intfloat/multilingual-e5-large-instruct", "intfloat/multilingual-e5-large-instruct"),
    "6": ("BGE-M3", None),
    "7": ("embed-english-v3.0", None),
}

QUERIES = [
    "Does the introduction of a Family Benefit Scheme by an employer completely extinguish a dependent's right to claim compassionate appointment?",
    "Can a person who has been convicted of a criminal offence and sentenced to imprisonment for more than two years be appointed as the Chief Minister of a State if their conviction has not been suspended?",
    "What was the Supreme Court's directive regarding the methods of recruitment for the Higher Judicial Service (District Judges) to ensure merit and efficiency?",
    "Do candidates placed on a waiting list have an absolute legal right to be appointed if the initially selected candidates fail to join the service?",
    "Are teachers employed in schools considered \"employees\" eligible for gratuity under Section 2(e) of the Payment of Gratuity Act, 1972?",
    "Under Section 319 of the Cr.P.C., can a trial court add new individuals as accused persons based merely on suspicion arising during the examination of witnesses?",
    "What procedure did the Supreme Court mandate for trial courts when objections are raised regarding the admissibility of documents during the evidence-recording stage?",
    "Can a criminal complaint under Section 138 of the Negotiable Instruments Act be maintained against a guarantor who issues a cheque solely to secure the debt of the principal debtor?",
    "Can criminal proceedings for cheating under Section 420/120B IPC continue against an assessee if their civil tax liability has already been fully settled under the Kar Vivad Samadhan Scheme, 1998?",
    "What degree of negligence is required to hold a medical professional criminally liable for the death of a patient under Section 304A of the IPC?"
]

GTS = [
    "No. The Supreme Court held that a Family Benefit Scheme (which provides a monthly deposit) cannot be equated with or replace the constitutional philosophy of social justice underlying compassionate appointments. The employer must still consider the dependent's application for compassionate employment.",
    "No. The Supreme Court ruled that a person disqualified from being a member of the legislature under Article 191(1)(e) read with Section 8(3) of the Representation of the People Act, 1951, due to a criminal conviction, cannot be legally appointed as Chief Minister, even if they enjoy the majority support of the legislative assembly.",
    "The Supreme Court directed that recruitment to the Higher Judicial Service should be divided into three avenues: 50% by promotion based on merit-cum-seniority, 25% by promotion strictly on merit through a limited departmental competitive examination, and 25% by direct recruitment from eligible advocates.",
    "No. The Supreme Court held that the existence of a waiting list does not create an indefeasible right to appointment. The employer has the discretion to carry forward unfilled vacancies to the next year, provided the decision is not arbitrary or mala fide.",
    "No. The Supreme Court held that teachers do not fall within the definition of \"employee\" under the Act because imparting education is a noble vocation and cannot be classified as skilled, unskilled, manual, supervisory, or clerical work.",
    "No. The Supreme Court held that the power under Section 319 is an extraordinary power that must be used sparingly. It requires a reasonable prospect of conviction and compelling reasons; mere suspicion is insufficient to subject a person to the agony of a criminal trial.",
    "To prevent unnecessary delays, the Supreme Court directed trial courts to tentatively mark the objected documents as exhibits and defer the final decision on their admissibility until the final judgment stage, rather than halting the trial to pass interlocutory orders.",
    "Yes. The Supreme Court ruled that the words \"any cheque\" and \"other liability\" in Section 138 are broad enough to cover cheques issued by a guarantor. The liability cannot be avoided merely because the cheque was issued as security for someone else's debt.",
    "No. The Supreme Court held that once the civil dispute is resolved and the authorities grant immunity under the Scheme, continuing the criminal prosecution lacks the requisite fraudulent intention and constitutes an abuse of the judicial process.",
    "The Supreme Court held that to secure a criminal conviction under Section 304A, the doctor's negligence must be \"gross\" or \"reckless.\" A mere lack of necessary care or an error of judgment, which might create civil liability in tort, is not sufficient for criminal punishment."
]

def run_primitive_pipeline(model_choice="1", api_key=None):
    import json

    if model_choice not in CHOICE_MAP:
        _write_status({"status": "error", "message": f"Invalid model choice: {model_choice}"})
        return None

    model_name, st_path = CHOICE_MAP[model_choice]
    _write_status({"status": "processing", "progress": f"Loading model: {model_name}..."})

    # Load embedder
    if model_choice == "6":
        try:
            from FlagEmbedding import BGEM3FlagModel
            class _BGE:
                def __init__(self): self.m = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
                def encode(self, texts, **kw): return self.m.encode(texts, batch_size=32, max_length=512, return_dense=True)["dense_vecs"]
            embedder = _BGE()
        except Exception as e:
            _write_status({"status": "error", "message": f"BGE-M3 load failed: {e}"}); return None
    elif model_choice == "7":
        from src.ingestion.embedder import CohereEmbedder
        embedder = CohereEmbedder()
    else:
        embedder = SentenceTransformer(st_path)

    print(f"[Primitive] Model loaded: {model_name}")
    _write_status({"status": "processing", "progress": "Parsing 516 PDFs (cached after first run)..."})

    chunks, pdf_files = load_all_chunks()

    dim = MODEL_DIMENSIONS[model_name]
    idx_name = MODEL_INDEX_MAP[model_name]
    _write_status({"status": "processing", "progress": f"Connecting to Pinecone index: {idx_name}..."})

    index = get_or_create_index(idx_name, dim)

    _write_status({"status": "processing", "progress": f"Embedding & upserting {len(chunks)} chunks to Qdrant...", "embedding": model_name})
    emb_time, index_size = embed_and_upsert(chunks, embedder, model_name, index)

    _write_status({"status": "processing", "progress": f"Running 10 queries via Groq + evaluating metrics...", "embedding": model_name})
    df = run_evaluation(model_name, embedder, index, chunks, QUERIES, GTS, emb_time, index_size)

    # Save CSV
    csv_path = BACKEND_DIR / f"results_{model_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Primitive] CSV saved → {csv_path}")

    # Build M-keyed summary
    col_map = {
        "M1_Embedding_Time":"M1","M2_Index_Size_Chunks":"M2","M3_Retrieval_Latency":"M3",
        "M4_Cosine_Similarity":"M4","M5_TopK_Accuracy":"M5","M6_ROUGE-1":"M6",
        "M7_ROUGE-2":"M7","M8_ROUGE-L":"M8","M9_Context_Length_Words":"M9",
        "M10_BLEU":"M10","M11_METEOR":"M11","M12_BERTScore_F1":"M12",
        "M13_Factual_Consistency_Deviation":"M13","M14_Faithfulness":"M14",
        "M15_Ground_Truth_Coverage":"M15","M16_End_to_End_Latency":"M16",
        "M17_Throughput_QPS":"M17","M18_CPU_Usage":"M18","M19_RAM_Usage_GB":"M19",
        "M20_Citation_Accuracy":"M20","M21_Terminology_Precision":"M21",
        "M22_Precedent_Coverage":"M22","M23_FCD_Score":"M23","M24_Bias_Score":"M24",
    }
    num_df = df.drop(columns=["GT_Reference","AI_Response"], errors="ignore")
    raw_means = num_df.mean().to_dict()
    summary = {col_map.get(k, k): v for k, v in raw_means.items()}

    report = {
        "status": "complete",
        "timestamp": time.ctime(),
        "progress": f"Primitive Audit Complete — {len(QUERIES)} queries · {len(pdf_files)} PDFs · {index_size} vectors",
        "summary": summary,
        "details": df.to_dict(orient="records"),
        "system_info": {
            "pipeline": "primitive",
            "vector_db": f"Pinecone ({idx_name})",
            "generator": "Groq llama-3.1-8b-instant",
            "judge": "Groq llama-3.1-8b-instant (M14,M15,M20-M24)",
            "embedding": model_name,
            "total_pdfs": len(pdf_files),
            "total_chunks": len(chunks),
        }
    }
    _write_status(report)
    print(f"[Primitive] Report saved → {STATUS_PATH}")
    return df

if __name__ == "__main__":
    run_primitive_pipeline(model_choice=input("Model choice (1/2/4/6): ").strip())
