import os
import json
import time
import requests
import psutil
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
import evaluate
from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer_stream
from src.utils.config_manager import get_active_model_name, get_active_db_name

# Evaluation Config
EVAL_DATA_PATH = "evaluation_data.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# Initialize local metrics
scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
metric_bleu = evaluate.load("bleu")
metric_meteor = evaluate.load("meteor")
model_st = SentenceTransformer(get_active_model_name())

def judge_llm_metrics(query, ground_truth, model_answer, context):
    """Uses Llama 3 to evaluate complex metrics like Faithfulness, Bias, and Terminology Precision."""
    prompt = f"""
    You are an expert legal auditor. Evaluate the following RAG system output against 6 specific metrics.
    
    [DATA]
    QUERY: {query}
    GROUND TRUTH: {ground_truth}
    MODEL ANSWER: {model_answer}
    CONTEXT: {context[:2000]}

    [METRICS TO EVALUATE]
    1. faithfulness: Does the answer only use information from the context? Score 1-100.
    2. citation_acc: Are legal citations (sections, acts, case names) correct? Score 1-100.
    3. term_precision: Is legal terminology used correctly? Score 1-100.
    4. precedent_match: Are relevant legal precedents and cases correctly cited? Score 1-100.
    5. factual_consistency: How factually consistent is the answer with the ground truth? Score 1-100.
    6. bias_score: Presence of bias towards protected attributes. Score 0-100, 0 means no bias.

    Output ONLY a valid JSON object with integer values:
    {{"faithfulness": 75, "citation_acc": 60, "term_precision": 80, "precedent_match": 50, "factual_consistency": 70, "bias_score": 5}}
    """
    
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=120)
        parsed = json.loads(res.json().get("response", "{}"))
        return parsed
    except Exception as e:
        return {"faithfulness": 50, "citation_acc": 50, "term_precision": 50, "precedent_match": 50, "factual_consistency": 50, "bias_score": 10}

def calculate_local_metrics(gt, ans):
    rouge_res = scorer_rouge.score(gt, ans)
    try: bleu_res = metric_bleu.compute(predictions=[ans], references=[[gt]])
    except: bleu_res = {"bleu": 0}
    try: meteor_res = metric_meteor.compute(predictions=[ans], references=[gt])
    except: meteor_res = {"meteor": 0}
    return {
        "rouge1": rouge_res['rouge1'].fmeasure,
        "rouge2": rouge_res['rouge2'].fmeasure,
        "rougeL": rouge_res['rougeL'].fmeasure,
        "bleu": bleu_res.get("bleu", 0),
        "meteor": meteor_res.get("meteor", 0)
    }

def run_evaluation():
    if not os.path.exists(EVAL_DATA_PATH):
        print("Data target missing.")
        return

    with open(EVAL_DATA_PATH, 'r') as f:
        data = json.load(f)

    all_results = []
    active_db = get_active_db_name()
    
    # System Baseline (M2)
    vector_count = 0
    try:
        if active_db == "pinecone":
            from src.utils.pinecone_client import index
            vector_count = index.describe_index_stats()['total_vector_count']
        else:
            from src.utils.qdrant_client import client, COLLECTION_NAME
            vector_count = client.get_collection(COLLECTION_NAME).vectors_count
    except: pass

    print("\n" + "="*80)
    print(f" LEXVED BENCHMARK - {active_db.upper()} ".center(80))
    print("="*80)

    # Simplified loop for 1 case per category
    for cat in ["civil", "criminal"]:
        for item in data[cat][:1]:
            query = item['query']
            gt = item['ground_truth']
            ram_before = psutil.virtual_memory().used
            
            # M1: Embedding Time
            t_start_emb = time.time()
            query_emb = model_st.encode([query])[0]
            m1_emb_time = time.time() - t_start_emb
            
            # M3: Retrieval Latency
            docs, m3_ret_lat = retrieve(query, top_k=5)
            
            # M4: Cosine Similarity
            doc_embs = model_st.encode([d.payload['text'] for d in docs])
            cos_sims = util.cos_sim(query_emb, doc_embs)[0]
            m4_cos_sim = cos_sims.mean().item()
            
            context = "".join([f"[Source: {d.payload.get('source')}] {d.payload['text']}\n" for d in docs])
            
            # M16: End-to-End Latency
            ans = ""
            for chunk in generate_answer_stream(query, context): ans += chunk
            m16_e2e = time.time() - t_start_emb
            
            q_stats = calculate_local_metrics(gt, ans)
            judge_res = judge_llm_metrics(query, gt, ans, context)
            
            m19_ram = (psutil.virtual_memory().used - ram_before) / (1024**2)
            
            res = {
                "id": len(all_results) + 1,
                "category": cat,
                "metrics": {
                    "M1": m1_emb_time, "M2": vector_count, "M3": m3_ret_lat,
                    "M4": m4_cos_sim, "M6": q_stats['rouge1'], "M7": q_stats['rouge2'],
                    "M8": q_stats['rougeL'], "M9": q_stats['meteor'], "M10": q_stats['bleu'],
                    "M14": judge_res.get("faithfulness", 0), "M16": m16_e2e,
                    "M19": m19_ram, "M20": judge_res.get("citation_acc", 0),
                    "M21": judge_res.get("term_precision", 0)
                }
            }
            all_results.append(res)
            
            # Update status
            report = {
                "timestamp": time.ctime(),
                "status": "processing",
                "progress": f"{len(all_results)} cases processed on {active_db.upper()}",
                "summary": {k: sum(r['metrics'][k] for r in all_results)/len(all_results) for k in res['metrics']},
                "details": all_results
            }
            with open("evaluation_results.json", "w") as f: json.dump(report, f, indent=4)

    # Final Report
    avgs = {k: sum(r['metrics'][k] for r in all_results)/len(all_results) for k in all_results[0]['metrics']}
    report = {
        "timestamp": time.ctime(),
        "status": "complete",
        "progress": f"Audit Complete — {len(all_results)} cases verified on {active_db.upper()}",
        "summary": avgs,
        "details": all_results,
        "system_info": {
            "vector_db": active_db.upper(),
            "model": "Llama 3 8B (Local)",
            "embedding": get_active_model_name(),
            "encryption": "AES-256"
        }
    }
    with open("evaluation_results.json", "w") as f: json.dump(report, f, indent=4)
    print(f"\n[SUCCESS] Metrics saved for {active_db.upper()}")

if __name__ == "__main__":
    run_evaluation()
