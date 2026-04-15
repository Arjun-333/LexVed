import os
import json
import time
import requests
import psutil
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func
import evaluate
from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer_stream
import qdrant_client

# Evaluation Config
EVAL_DATA_PATH = "evaluation_data.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# Initialize local metrics
scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
metric_bleu = evaluate.load("bleu")
metric_meteor = evaluate.load("meteor")
model_st = SentenceTransformer('all-mpnet-base-v2')

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
    1. Faithfulness (M14): Does the answer only use context? (1-100 scale)
    2. Citation Accuracy (M20): Are legal citations correct? (1-100 scale)
    3. Terminology Precision (M21): Is legal jargon used correctly? (1-100 scale)
    4. Precedent Coverage (M22): Are multiple relevant cases retrieved where needed? (1-100 scale)
    5. Factual Consistency Deviation (M13): Hallucination level (0-100, 0 is best)
    6. Bias Score (M24): Presence of protected attributes (0-100, 0 is best)

    Output ONLY a JSON object: 
    {{
        "faithfulness": <int>,
        "citation_acc": <int>,
        "term_precision": <int>,
        "precedent_cov": <int>,
        "fcd_score": <int>,
        "bias_score": <int>,
        "reason": "<one_sentence_summary>"
    }}
    """
    
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=45)
        return json.loads(res.json().get("response", "{}"))
    except:
        return {}

def calculate_local_metrics(gt, ans):
    """Calculates ROUGE, BLEU, METEOR, and BERTScore."""
    # ROUGE
    rouge_res = scorer_rouge.score(gt, ans)
    
    # BLEU
    try:
        bleu_res = metric_bleu.compute(predictions=[ans], references=[[gt]])
    except:
        bleu_res = {"bleu": 0}
        
    # METEOR
    try:
        meteor_res = metric_meteor.compute(predictions=[ans], references=[gt])
    except:
        meteor_res = {"meteor": 0}

    # BERTScore
    P, R, F1 = bert_score_func([ans], [gt], lang="en", verbose=False)
    
    return {
        "rouge1": rouge_res['rouge1'].fmeasure,
        "rouge2": rouge_res['rouge2'].fmeasure,
        "rougeL": rouge_res['rougeL'].fmeasure,
        "bleu": bleu_res.get("bleu", 0),
        "meteor": meteor_res.get("meteor", 0),
        "bertscore": F1.item()
    }

def run_evaluation():
    if not os.path.exists(EVAL_DATA_PATH):
        print("Data target missing.")
        return

    with open(EVAL_DATA_PATH, 'r') as f:
        data = json.load(f)

    all_results = []
    
    # System Baseline (M2)
    q_client = qdrant_client.QdrantClient("localhost", port=6333)
    try:
        vector_count = q_client.count("lexved_chunks").count
    except:
        vector_count = 0

    print("\n" + "="*80)
    print(" LEXVED MISSION-CRITICAL BENCHMARK (M1-M24) ".center(80))
    print("="*80)

    for cat in ["civil", "criminal"]:
        for item in data[cat][:1]: # Only run 1 sample per category for fast verification
            query = item['query']
            gt = item['ground_truth']
            
            # Start Tracking Metrics
            cpu_before = psutil.cpu_percent()
            ram_before = psutil.virtual_memory().used
            
            # M1: Embedding Time (Mock search query)
            t_start_emb = time.time()
            query_emb = model_st.encode([query])[0]
            m1_emb_time = time.time() - t_start_emb
            
            # M3: Retrieval Latency
            t_start_ret = time.time()
            docs, m3_ret_lat = retrieve(query, top_k=5)
            
            # M4: Cosine Similarity
            doc_embs = model_st.encode([d.payload['text'] for d in docs])
            cos_sims = util.cos_sim(query_emb, doc_embs)[0]
            m4_cos_sim = cos_sims.mean().item()
            
            context = ""
            for d in docs: context += f"[Source: {d.payload.get('source')}] {d.payload['text']}\n"
            
            # M16: End-to-End Latency (Start Gen)
            t_start_gen = time.time()
            ans = ""
            for chunk in generate_answer_stream(query, context): ans += chunk
            m16_e2e = time.time() - t_start_ret # Total time from start of retrieval
            
            # Capture System Usage (M18, M19)
            m18_cpu = psutil.cpu_percent()
            m19_ram = (psutil.virtual_memory().used - ram_before) / (1024**2) # MB delta
            
            # Quality Metrics (M6-M12, M15)
            q_stats = calculate_local_metrics(gt, ans)
            
            # LLM Judge Metrics (M13, M14, M20, M21, M22, M24)
            judge_res = judge_llm_metrics(query, gt, ans, context)
            
            # Store result
            res = {
                "id": len(all_results) + 1,
                "category": cat,
                "metrics": {
                    "M1": m1_emb_time,
                    "M2": vector_count,
                    "M3": m3_ret_lat,
                    "M4": m4_cos_sim,
                    "M6": q_stats['rouge1'],
                    "M10": q_stats['bleu'],
                    "M12": q_stats['bertscore'],
                    "M14": judge_res.get("faithfulness", 0),
                    "M16": m16_e2e,
                    "M18": m18_cpu,
                    "M20": judge_res.get("citation_acc", 0),
                    "M21": judge_res.get("term_precision", 0)
                }
            }
            all_results.append(res)
            print(f"[{cat.upper()}] Case {res['id']}: Score(BERT)={q_stats['bertscore']:.2f} | Latency={m16_e2e:.2f}s")

    # Output Final Table
    print("\n" + "-"*80)
    print(" METRICS EVALUATED VALUE REPORT ".center(80))
    print("-"*80)
    headers = ["MetricID", "Category", "Result", "Unit"]
    print(f"{headers[0]:<10} | {headers[1]:<25} | {headers[2]:<15} | {headers[3]:<10}")
    print("-"*80)
    
    # Aggregated results for one representative case (Summary)
    final = all_results[0]['metrics'] # Use first case for summary display
    avgs = {k: sum(r['metrics'][k] for r in all_results)/len(all_results) for k in final if isinstance(final[k], (int, float))}
    
    print(f"M1         | Retrieval Performance     | {avgs['M1']:.4f}        | sec")
    print(f"M2         | Retrieval Performance     | {avgs['M2']}             | vectors")
    print(f"M3         | Retrieval Performance     | {avgs['M3']:.4f}        | sec")
    print(f"M4         | Retrieval Performance     | {avgs['M4']:.4f}        | score")
    print(f"M6         | Answer Quality            | {avgs['M6']:.4f}        | score")
    print(f"M10        | Answer Quality            | {avgs['M10']:.4f}        | score")
    print(f"M12        | Answer Quality            | {avgs['M12']:.4f}        | score")
    print(f"M14        | Answer Quality            | {avgs['M14']}%           | %")
    print(f"M16        | System Efficiency         | {avgs['M16']:.2f}        | sec")
    print(f"M20        | Legal-Specific            | {avgs['M20']}%           | %")
    print(f"M21        | Legal-Specific            | {avgs['M21']}%           | %")
    print("-"*80)

if __name__ == "__main__":
    run_evaluation()
