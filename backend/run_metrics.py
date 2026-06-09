"""
LexVed Enhanced Pipeline Evaluation Runner
References the same methodology as LexVed_Institutional_Audit.ipynb:
  - Hybrid BM25/Dense retrieval with Reciprocal Rank Fusion
  - CrossEncoder reranking (ms-marco-MiniLM-L-6-v2)
  - LLM Judge via Groq (llama-3.1-8b-instant)
  - 26 KPI metrics (M1-M26)

Writes results to evaluation_results.json for the frontend dashboard.
"""
import os
import json
import time
import re
import psutil
import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation_queries.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "evaluation_results.json")

# ─── Groq LLM Judge (from notebook) ──────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _update_progress(msg, status="processing"):
    """Write progress to evaluation_results.json so the frontend can poll."""
    try:
        existing = {}
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r") as f:
                existing = json.load(f)
        existing["status"] = status
        existing["progress"] = msg
        existing["pid"] = os.getpid()
        with open(RESULTS_PATH, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        pass


def unified_judge(query, context, answer, ground_truth) -> dict:
    """LLM-based judge matching the notebook's unified_judge function."""
    import requests

    defaults = {
        "faithfulness": 50, "citation_acc": 50, "term_precision": 50,
        "precedent_match": 50, "factual_consistency": 50, "bias_score": 10,
        "regulatory_alignment": 50, "jurisdictional_comp": 50
    }
    prompt = f"""You are an expert legal auditor. Evaluate the RAG output on 8 metrics.
QUERY: {query}
GROUND TRUTH: {ground_truth}
MODEL ANSWER: {answer}
CONTEXT: {context[:5000]}

Return ONLY valid JSON with integer scores 0-100:
{{"faithfulness":75,"citation_acc":60,"term_precision":80,"precedent_match":50,"factual_consistency":70,"bias_score":5,"regulatory_alignment":85,"jurisdictional_comp":90}}"""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(5):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                raw = r.json()["choices"][0]["message"]["content"]
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
                    return {**defaults, **{k.lower(): v for k, v in parsed.items()}}
            elif r.status_code == 429:
                print("[LexVed] Judge rate limit (429). Waiting 30s...")
                time.sleep(30)
            else:
                time.sleep(5)
        except Exception:
            time.sleep(5)
    return defaults


def _jval(judge, key, default=50.0):
    try:
        return float(str(judge.get(key, default)).strip()) / 100.0
    except Exception:
        return default / 100.0


def generate_llm_answer(query, context):
    """Generate an answer using the active generation model via the project's generator.
    Also returns (answer, prefill_latency, ttft, throughput) by streaming the response.
    """
    import requests
    import tiktoken

    prompt = f"""Answer the following query using ONLY the context below. Keep it professional.
Context:
{context}

Query: {query}
Answer:"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "Groq-Beta": "inference-metrics"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "stream": True,
        "stream_options": {
            "include_usage": True
        }
    }
    for attempt in range(5):
        try:
            t_start = time.time()
            r = requests.post(GROQ_URL, headers=headers, json=payload, stream=True, timeout=60)
            if r.status_code == 200:
                answer = ""
                ttft = 0.0
                prefill_latency = 0.0
                throughput = 0.0
                first_token_received = False
                first_token_time = None
                for line in r.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:]
                        if data_content == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_content)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    if not first_token_received:
                                        first_token_time = time.time()
                                        ttft = first_token_time - t_start
                                        first_token_received = True
                                    answer += content
                            
                            usage = chunk.get("usage") or chunk.get("x_groq", {}).get("usage")
                            if usage:
                                prefill_latency = usage.get("prompt_time", 0.0)
                                completion_tokens = usage.get("completion_tokens", 0)
                                completion_time = usage.get("completion_time", 0.0)
                                if completion_time > 0:
                                    throughput = completion_tokens / completion_time
                        except Exception:
                            pass
                t_end = time.time()
                if answer and ttft == 0.0:
                    ttft = t_end - t_start
                if prefill_latency == 0.0:
                    prefill_latency = ttft
                if throughput == 0.0:
                    try:
                        enc = tiktoken.get_encoding("cl100k_base")
                    except Exception:
                        class FakeEnc:
                            def encode(self, text):
                                return text.split()
                        enc = FakeEnc()
                    ans_tokens = len(enc.encode(answer))
                    gen_time = t_end - (first_token_time or t_start)
                    if gen_time > 0:
                        throughput = ans_tokens / gen_time
                return answer, prefill_latency, ttft, throughput
            elif r.status_code == 429:
                print("[LexVed] Generation rate limit (429). Waiting 30s...")
                time.sleep(30)
            else:
                time.sleep(5)
        except Exception as e:
            print(f"[LexVed] Error in generation attempt {attempt}: {e}")
            time.sleep(5)
    return "", 0.0, 0.0, 0.0



def run_evaluation():
    """Full 24-KPI evaluation using the Enhanced Pipeline (matching the notebook)."""
    from src.utils.config_manager import get_active_model_name, get_active_db_name
    from src.retrieval.retriever import retrieve
    from src.ingestion.embedder import get_embeddings

    active_model = get_active_model_name()
    active_db = get_active_db_name()
    print(f"\n[LexVed] >>> STARTING ENHANCED PIPELINE AUDIT: {active_model} on {active_db} <<<")

    _update_progress(f"Loading evaluation queries for {active_model}...")

    # Load evaluation data
    if not os.path.exists(EVAL_DATA_PATH):
        # Fallback to root-level evaluation_data.json
        alt_path = os.path.join(PROJECT_ROOT, "evaluation_data.json")
        if os.path.exists(alt_path):
            eval_path = alt_path
        else:
            _update_progress("No evaluation data found.", status="error")
            return
    else:
        eval_path = EVAL_DATA_PATH

    with open(eval_path, "r") as f:
        data = json.load(f)

    queries = []
    ground_truths = []
    for cat in ["civil", "criminal"]:
        for item in data.get(cat, []):
            queries.append(item["query"])
            ground_truths.append(item["ground_truth"])

    total_queries = len(queries)
    if total_queries == 0:
        _update_progress("No queries found in evaluation data.", status="error")
        return

    print(f"[LexVed] Loaded {total_queries} evaluation queries.")

    # --- Compute M1 (Embedding Latency) & M2 (Index Size) ---
    _update_progress("Computing embedding latency (M1)...")
    t_emb = time.time()
    _ = get_embeddings(["benchmark latency test query"])
    emb_latency = time.time() - t_emb

    # Get index size from active DB
    index_size = 0
    try:
        if active_db == "qdrant":
            from qdrant_client import QdrantClient
            from src.utils.qdrant_provider import COLLECTION_NAME
            client = QdrantClient(host="localhost", port=6333)
            info = client.get_collection(COLLECTION_NAME)
            index_size = info.points_count
        else:
            from src.utils.pinecone_client import index as pc_index
            stats = pc_index.describe_index_stats()
            index_size = stats.get("total_vector_count", 0)
    except Exception as e:
        print(f"[LexVed] Could not get index size: {e}")

    # --- Per-Query Evaluation (Enhanced Pipeline from notebook) ---
    preds = []
    ret_texts_all = []
    q_vecs = []
    r_times = []
    g_times = []
    prefill_latencies = []
    ttft_latencies = []
    throughput_rates = []

    for i, query in enumerate(queries):
        _update_progress(f"Evaluating query {i+1}/{total_queries}: {query[:50]}...")
        print(f"[Auditor] Query {i+1}/{total_queries}: {query[:60]}...")

        t_start = time.time()
        q_vec = get_embeddings([query])[0]

        # Enhanced retrieval (Hybrid BM25 + Dense + RRF + CrossEncoder)
        docs, ret_time = retrieve(query, top_k=5)
        r_times.append(ret_time)

        ret = [d.payload.get("text", "") for d in docs]
        context_str = "\n\n".join(ret)

        # Generation via LLM
        t1 = time.time()
        ans, prefill_lat, ttft, throughput = generate_llm_answer(query, context_str)
        g_times.append(time.time() - t1)
        prefill_latencies.append(prefill_lat)
        ttft_latencies.append(ttft)
        throughput_rates.append(throughput)

        preds.append(ans)
        ret_texts_all.append(ret)
        q_vecs.append(q_vec)

    # --- Batch metrics computation (matching notebook) ---
    _update_progress("Computing NLP metrics (ROUGE, BLEU, METEOR, BERTScore)...")

    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        for r in ['wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
            nltk.download(r, quiet=True)
    except Exception:
        meteor_score = None

    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        bert_score_fn = None

    from sklearn.metrics.pairwise import cosine_similarity

    # BERTScore (batched)
    bert_f1_scores = [0.0] * len(preds)
    bert_ctx_scores = [0.0] * len(preds)
    if bert_score_fn:
        try:
            _, _, F1_gt = bert_score_fn(preds, ground_truths, lang='en', verbose=False)
            bert_f1_scores = F1_gt.tolist()
        except Exception:
            pass

        contexts_joined = [" ".join(ret_texts_all[i]) for i in range(len(preds))]
        try:
            _, _, F1_ctx = bert_score_fn(preds, contexts_joined, lang='en', verbose=False)
            bert_ctx_scores = F1_ctx.tolist()
        except Exception:
            pass
    else:
        contexts_joined = [" ".join(ret_texts_all[i]) for i in range(len(preds))]

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    metric_rows = []

    for i in range(len(preds)):
        _update_progress(f"Computing metrics for query {i+1}/{total_queries}...")

        r_scores = rouge.score(ground_truths[i], preds[i])
        bert_f1 = bert_f1_scores[i]
        fcd = float(1 - bert_ctx_scores[i])

        try:
            bleu = sentence_bleu([ground_truths[i].split()], preds[i].split(), smoothing_function=smoothie)
        except Exception:
            bleu = 0.0
        try:
            met = meteor_score([ground_truths[i].split()], preds[i].split()) if meteor_score else 0.0
        except Exception:
            met = 0.0

        ctx_vecs = get_embeddings(ret_texts_all[i]) if ret_texts_all[i] else np.zeros((1, 1))
        cosine_sim = float(np.mean(cosine_similarity([q_vecs[i]], ctx_vecs))) if len(ctx_vecs) else 0.0
        sims_arr = cosine_similarity([q_vecs[i]], ctx_vecs)[0] if len(ctx_vecs) else []
        topk_acc = float(np.mean([1 if s > 0.8 else 0 for s in sims_arr])) if len(sims_arr) else 0.0

        # LLM Judge (8 legal KPIs)
        _update_progress(f"LLM Judge evaluating query {i+1}/{total_queries}...")
        judge = unified_judge(queries[i], contexts_joined[i], preds[i], ground_truths[i])

        e2e = r_times[i] + g_times[i]

        metric_rows.append({
            "M3": r_times[i],
            "M4": cosine_sim,
            "M5": topk_acc,
            "M6": r_scores["rouge1"].fmeasure,
            "M7": r_scores["rouge2"].fmeasure,
            "M8": r_scores["rougeL"].fmeasure,
            "M9": len(contexts_joined[i].split()),
            "M10": bleu,
            "M11": met,
            "M12": bert_f1,
            "M13": fcd,
            "M14": _jval(judge, "faithfulness"),
            "M15": _jval(judge, "factual_consistency") * 100,
            "M16": e2e,
            "M17": round(1.0 / max(0.001, e2e), 4),
            "M18": psutil.cpu_percent(),
            "M19": round(psutil.virtual_memory().used / (1024**3), 2),
            "M20": _jval(judge, "citation_acc"),
            "M21": _jval(judge, "term_precision"),
            "M22": _jval(judge, "precedent_match") * 100,
            "M23": _jval(judge, "regulatory_alignment"),
            "M24": _jval(judge, "bias_score"),
            "M25": ttft_latencies[i],
            "M26": prefill_latencies[i],
            "M27": throughput_rates[i],
        })

    # --- Aggregate all metrics (mean across queries) ---
    import pandas as pd
    df = pd.DataFrame(metric_rows)
    summary = df.mean().to_dict()
    summary["M1"] = emb_latency
    summary["M2"] = index_size

    # --- Count actual PDF docs in corpus ---
    corpus_dir = os.path.join(PROJECT_ROOT, "data", "PDF")
    total_docs = 0
    if os.path.exists(corpus_dir):
        for root, dirs, files in os.walk(corpus_dir):
            for f in files:
                if f.lower().endswith(".pdf"):
                    total_docs += 1

    # --- Build detailed results per query ---
    details = []
    for i in range(len(queries)):
        details.append({
            "query": queries[i],
            "ground_truth": ground_truths[i],
            "answer": preds[i],
            "metrics": metric_rows[i]
        })

    # --- Write final report ---
    report = {
        "status": "done",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "vector_db": active_db,
            "embedding": active_model,
            "model": "llama-3.1-8b-instant",
            "total_documents": total_docs,
            "index_size": index_size,
            "pipeline": "enhanced"
        },
        "summary": summary,
        "details": details,
        "pid": os.getpid()
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[LexVed] Audit for {active_model} complete. Results saved to {RESULTS_PATH}")
    return report


if __name__ == "__main__":
    run_evaluation()
