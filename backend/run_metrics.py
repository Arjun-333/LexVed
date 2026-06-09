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
        "statements": ["The model answer is factually consistent."],
        "supported": [True],
        "citation_acc": 50, "term_precision": 50, "precedent_match": 50,
        "bias_score": 10, "regulatory_alignment": 50, "jurisdictional_comp": 50
    }
    prompt = f"""You are an expert legal auditor. Evaluate the RAG output's Faithfulness and check citation and precedent details.
QUERY: {query}
GROUND TRUTH: {ground_truth}
MODEL ANSWER: {answer}
CONTEXT: {context[:5000]}

Please perform the following audit steps:
1. Break down the MODEL ANSWER into individual factual statements.
2. For each statement, verify if it is directly supported by the CONTEXT (Yes/No).
3. Evaluate other metrics on a scale of 0-100:
   - citation_acc: Accuracy of citations used.
   - term_precision: Precision of legal terminology.
   - precedent_match: Alignment with legal precedents.
   - bias_score: Presence of bias or subjectivity (0 is best, 100 is worst).
   - regulatory_alignment: Alignment with regulations.
   - jurisdictional_comp: Jurisdictional competence.

Return ONLY valid JSON:
{{
  "statements": ["statement 1", "statement 2"],
  "supported": [true, false],
  "citation_acc": 80,
  "term_precision": 90,
  "precedent_match": 75,
  "bias_score": 5,
  "regulatory_alignment": 85,
  "jurisdictional_comp": 90
}}"""

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
                    # Handle key case-insensitivity
                    norm_parsed = {}
                    for k, v in parsed.items():
                        norm_parsed[k.lower()] = v
                    return {**defaults, **norm_parsed}
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
    synthetic_path = os.path.join(PROJECT_ROOT, "data", "synthetic_evaluation_dataset.json")
    gold_chunks = []
    
    if os.path.exists(synthetic_path):
        _update_progress(f"Loading synthetic dataset from {synthetic_path}...")
        with open(synthetic_path, "r") as f:
            synthetic_data = json.load(f)
        queries = [item["query"] for item in synthetic_data]
        ground_truths = [item["ground_truth"] for item in synthetic_data]
        gold_chunks = [item["gold_chunk_text"] for item in synthetic_data]
    else:
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
        gold_chunks = [None] * len(queries)

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
    all_ret_texts_all = []
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
        docs, ret_time = retrieve(query, top_k=10)
        r_times.append(ret_time)

        ret_all = [d.payload.get("text", "") for d in docs]
        ret_5 = ret_all[:5]
        context_str = "\n\n".join(ret_5)

        # Generation via LLM
        t1 = time.time()
        ans, prefill_lat, ttft, throughput = generate_llm_answer(query, context_str)
        g_times.append(time.time() - t1)
        prefill_latencies.append(prefill_lat)
        ttft_latencies.append(ttft)
        throughput_rates.append(throughput)

        preds.append(ans)
        ret_texts_all.append(ret_5)
        all_ret_texts_all.append(ret_all)
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

    from nltk.stem import PorterStemmer
    ps = PorterStemmer()

    def is_match(ret_doc, gold_doc):
        if not ret_doc or not gold_doc:
            return False
        # 1. Normalize spaces & lowercase
        r_norm = "".join(ret_doc.split()).lower()
        g_norm = "".join(gold_doc.split()).lower()
        if g_norm in r_norm or r_norm in g_norm:
            return True
        # 2. Token overlap checks
        r_toks = set(ret_doc.lower().split())
        g_toks = set(gold_doc.lower().split())
        if not r_toks or not g_toks:
            return False
        overlap_ratio = len(r_toks & g_toks) / len(g_toks)
        if overlap_ratio >= 0.60:
            return True
        jaccard = len(r_toks & g_toks) / len(r_toks | g_toks)
        return jaccard >= 0.60

    def verify_citations(pred_text, gt_text):
        pattern = r'(Section\s+\d+[A-Za-z]*|S\.\s*\d+|Article\s+\d+|Art\.\s*\d+|Act,\s+\d{4}|[A-Z]{3,4}\s+\d{4}\s+[A-Z\s]+|AIR\s+\d{4}\s+SC\s+\d+|\(\d{4}\)\s+\d+\s+SCC\s+\d+)'
        gt_citations = set(re.findall(pattern, gt_text, re.IGNORECASE))
        if not gt_citations:
            return None
        pred_citations = set(re.findall(pattern, pred_text, re.IGNORECASE))
        matched = gt_citations & pred_citations
        return len(matched) / len(gt_citations)

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    metric_rows = []

    for i in range(len(preds)):
        _update_progress(f"Computing metrics for query {i+1}/{total_queries}...")

        r_scores = rouge.score(ground_truths[i], preds[i])
        bert_f1 = bert_f1_scores[i]

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
        
        # Retrieve target gold document if available (non-circular evaluation)
        gold_chunk = gold_chunks[i] if (i < len(gold_chunks) and gold_chunks[i] is not None) else None
        
        # Determine the retrieved texts
        ret_texts_5 = ret_texts_all[i]
        ret_texts_10 = all_ret_texts_all[i] if i < len(all_ret_texts_all) else ret_texts_5

        # Check robust matches
        matches_5 = [doc for doc in ret_texts_5 if is_match(doc, gold_chunk)]
        matches_10 = [doc for doc in ret_texts_10 if is_match(doc, gold_chunk)]
        has_gold = (gold_chunk is not None)

        recall_at_5 = 1.0 if (has_gold and matches_5) else 0.0
        recall_at_10 = 1.0 if (has_gold and matches_10) else 0.0
        precision_at_5 = len(matches_5) / 5.0
        
        # MRR
        mrr = 0.0
        for rank, doc in enumerate(ret_texts_10):
            if is_match(doc, gold_chunk):
                mrr = 1.0 / (rank + 1)
                break
        
        # nDCG@10
        ndcg_at_10 = 0.0
        for rank, doc in enumerate(ret_texts_10):
            if is_match(doc, gold_chunk):
                ndcg_at_10 = 1.0 / np.log2(rank + 2)
                break

        # Real Ground Truth Coverage (M15) using Porter Stemmer
        cleaned_gt = re.sub(r'[^\w\s]', '', ground_truths[i].lower())
        cleaned_ctx = re.sub(r'[^\w\s]', '', contexts_joined[i].lower())
        gt_tokens = {ps.stem(w) for w in cleaned_gt.split()}
        ctx_tokens = {ps.stem(w) for w in cleaned_ctx.split()}
        gt_coverage = len(gt_tokens & ctx_tokens) / max(1, len(gt_tokens))

        # LLM Judge (8 legal KPIs)
        _update_progress(f"LLM Judge evaluating query {i+1}/{total_queries}...")
        judge = unified_judge(queries[i], contexts_joined[i], preds[i], ground_truths[i])

        # RAGAS-like Faithfulness (M14)
        statements = judge.get("statements", ["The model answer is factually consistent."])
        supported = judge.get("supported", [True])
        if not isinstance(statements, list) or not isinstance(supported, list) or len(statements) != len(supported) or not statements:
            faithfulness_score = 0.5
        else:
            faithfulness_score = sum(1 for x in supported if x) / len(supported)

        # Factual Consistency Deviation (M13)
        fcd = 1.0 - faithfulness_score

        # Objective Regex Citation Accuracy (M20)
        citation_acc = verify_citations(preds[i], ground_truths[i])

        e2e = r_times[i] + g_times[i]

        metric_rows.append({
            "M3": r_times[i],
            "M4": cosine_sim,
            "M5": recall_at_5,
            "M6": r_scores["rouge1"].fmeasure,
            "M7": r_scores["rouge2"].fmeasure,
            "M8": r_scores["rougeL"].fmeasure,
            "M9": len(contexts_joined[i].split()),
            "M10": bleu,
            "M11": met,
            "M12": bert_f1,
            "M13": fcd,
            "M14": faithfulness_score,
            "M15": gt_coverage * 100,
            "M16": e2e,
            "M17": round(1.0 / max(0.001, e2e), 4),
            "M18": psutil.Process(os.getpid()).cpu_percent(),
            "M19": round(psutil.Process(os.getpid()).memory_info().rss / (1024**3), 2),
            "M20": citation_acc * 100 if citation_acc is not None else None,
            "M21": _jval(judge, "term_precision"),
            "M22": _jval(judge, "precedent_match") * 100,
            "M23": _jval(judge, "regulatory_alignment"),
            "M24": _jval(judge, "bias_score"),
            "M25": ttft_latencies[i],
            "M26": prefill_latencies[i],
            "M27": throughput_rates[i],
            "M28": recall_at_10,
            "M29": mrr,
            "M30": ndcg_at_10,
            "M31": precision_at_5,
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
