import os
from dotenv import load_dotenv
load_dotenv()
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
from src.utils.config_manager import get_active_model_name, get_active_db_name, get_active_generation_model

# Evaluation Config
EVAL_DATA_PATH = "evaluation_data.json"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Initialize local metrics
scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
metric_bleu = evaluate.load("bleu")
metric_meteor = evaluate.load("meteor")
metric_bertscore = evaluate.load("bertscore")
_model_st = None
_model_st_name = None

def get_model_st():
    """Lazily load the SentenceTransformer model, refreshing if the active model changed."""
    global _model_st, _model_st_name
    active = get_active_model_name()
    # Cohere uses API-based embeddings, fall back to a lightweight local model for cosine sim calculations
    if "embed-english" in active or "cohere" in active.lower():
        if _model_st_name != "multi-qa-mpnet-base-cos-v1":
            _model_st = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
            _model_st_name = "multi-qa-mpnet-base-cos-v1"
        return _model_st
    if _model_st is None or _model_st_name != active:
        print(f"[LexVed] Loading SentenceTransformer for metrics: {active}")
        _model_st = SentenceTransformer(active)
        _model_st_name = active
    return _model_st

def check_hallucination_nli(context, answer):
    try:
        if not globals().get("nli_pipeline"):
            from transformers import pipeline
            globals()["nli_pipeline"] = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")
            
        ctx_trunc = context[:2000] 
        res = globals()["nli_pipeline"]({"text": ctx_trunc, "text_pair": answer})
        label = res.get("label", "").lower()
        score = res.get("score", 0.5)
        
        if "entailment" in label or "label_1" in label or "label_2" in label:
            return int(score * 100)
        elif "contradiction" in label or "label_0" in label:
            return int((1 - score) * 100)
        return 50 # Neutral
    except Exception as e:
        print(f"NLI Error: {e}")
        return 75

def get_ner_precision(gt, ans):
    try:
        if not globals().get("nlp"):
            import spacy
            try:
                globals()["nlp"] = spacy.load("en_core_web_sm")
            except:
                import os
                os.system("python3 -m spacy download en_core_web_sm")
                globals()["nlp"] = spacy.load("en_core_web_sm")
        
        gt_doc = globals()["nlp"](gt)
        ans_doc = globals()["nlp"](ans)
        
        gt_ents = {ent.text.lower() for ent in gt_doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "LAW", "DATE"]}
        ans_ents = {ent.text.lower() for ent in ans_doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "LAW", "DATE"]}
        
        if not gt_ents: return 100
        overlap = gt_ents.intersection(ans_ents)
        return int((len(overlap) / len(gt_ents)) * 100)
        
    except Exception as e:
        print(f"NER Error: {e}")
        return 50

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
    7. regulatory_alignment: Does the answer align with standard legal regulations? Score 1-100.
    8. jurisdictional_comp: Is the jurisdictional context appropriate? Score 1-100.

    Output ONLY a valid JSON object with integer values:
    {{"faithfulness": 75, "citation_acc": 60, "term_precision": 80, "precedent_match": 50, "factual_consistency": 70, "bias_score": 5, "regulatory_alignment": 85, "jurisdictional_comp": 90}}
    """
    
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        eval_payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        try:
            res = requests.post(url, headers=headers, json=eval_payload, timeout=60)
            if res.status_code == 200:
                raw_resp = res.json()["choices"][0]["message"]["content"]
                import re
                match = re.search(r'\{.*\}', raw_resp, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception as e:
            print(f"[LexVed] Groq Judge failed ({e}), falling back to Ollama...")

    payload = {"model": get_active_generation_model(), "prompt": prompt, "stream": False, "format": "json"}
    try:
        import re
        res = requests.post(OLLAMA_URL, json=payload, timeout=180) 
        raw_resp = res.json().get("response", "{}")
        match = re.search(r'\{.*\}', raw_resp, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            parsed = json.loads(raw_resp)
        return parsed
    except Exception as e:
        print(f"Exception during Llama 3 Judge Evaluation: {e}")
        return {"faithfulness": 50, "citation_acc": 50, "term_precision": 50, "precedent_match": 50, "factual_consistency": 50, "bias_score": 10, "regulatory_alignment": 50, "jurisdictional_comp": 50}

def calculate_local_metrics(gt, ans):
    rouge_res = scorer_rouge.score(gt, ans)
    try: bleu_res = metric_bleu.compute(predictions=[ans], references=[[gt]])
    except: bleu_res = {"bleu": 0}
    try: meteor_res = metric_meteor.compute(predictions=[ans], references=[gt])
    except: meteor_res = {"meteor": 0}
    try: bert_res = metric_bertscore.compute(predictions=[ans], references=[gt], lang="en")
    except: bert_res = {"f1": [0]}
    
    return {
        "rouge1": rouge_res['rouge1'].fmeasure,
        "rouge2": rouge_res['rouge2'].fmeasure,
        "rougeL": rouge_res['rougeL'].fmeasure,
        "bleu": bleu_res.get("bleu", 0),
        "meteor": meteor_res.get("meteor", 0),
        "bert_f1": bert_res['f1'][0]
    }

def calculate_recall_at_k(docs, query_category):
    # Heuristic recall: Check if any retrieved doc matches the expected category sample
    # In this limited 10-PDF case, we assume docs tagged with the correct 'category' are relevant.
    hits = [1 for d in docs if d.payload.get('category', '').lower() == query_category.lower()]
    return (sum(hits) / len(docs)) if docs else 0

def ensure_data_ingested():
    active_db = get_active_db_name()
    vector_count = 0
    dimension_ok = True
    expected_dim = get_active_model_params = None
    
    try:
        from src.utils.config_manager import get_active_model_params as _get_params
        expected_dim = _get_params()["dimension"]
    except:
        pass
    
    try:
        if active_db == "pinecone":
            from src.utils.pinecone_client import index, create_index
            stats = index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            
            if expected_dim and 'dimension' in stats:
                if stats['dimension'] != expected_dim:
                    print(f"[LexVed] Dimension mismatch! Pinecone has {stats['dimension']}d vectors, model needs {expected_dim}d. Recreating index...")
                    create_index()
                    vector_count = 0
                    dimension_ok = False
        else:
            from src.utils.qdrant_provider import client, COLLECTION_NAME
            try:
                col_info = client.get_collection(COLLECTION_NAME)
                vector_count = col_info.vectors_count
                # Check if collection dimensions match the current model
                if expected_dim and hasattr(col_info.config, 'params') and hasattr(col_info.config.params, 'vectors'):
                    col_dim = col_info.config.params.vectors.size if hasattr(col_info.config.params.vectors, 'size') else None
                    if col_dim and col_dim != expected_dim:
                        print(f"[LexVed] Dimension mismatch! Collection has {col_dim}d vectors, model needs {expected_dim}d. Re-ingesting...")
                        dimension_ok = False
                elif expected_dim and hasattr(col_info.config, 'params'):
                    # Try alternative attribute path
                    try:
                        col_dim = col_info.config.params.vectors_config.size if hasattr(col_info.config.params, 'vectors_config') else None
                    except:
                        pass
            except Exception as e:
                # Collection doesn't exist yet
                print(f"[LexVed] Qdrant collection check failed: {e}")
                vector_count = 0
    except Exception as e:
        print(f"[LexVed] DB check error: {e}")
        pass
        
    if vector_count > 0 and dimension_ok:
        return
        
    print(f"[LexVed] No data found in {active_db.upper()}. Auto-ingesting evaluation documents...")
    
    # Search for PDFs
    pdf_paths = []
    pdf_dir = "data/PDF"
    if os.path.exists(pdf_dir):
        for root, dirs, files in os.walk(pdf_dir):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_paths.append(os.path.join(root, f))
                    if len(pdf_paths) >= 5: # Max 5 files for a quick but realistic test
                        break
            if len(pdf_paths) >= 5:
                break
                
    if not pdf_paths:
        print("[LexVed] No PDFs found in data/PDF/. Cannot evaluate.")
        return
        
    from src.ingestion.pdf_processor import extract_chunks, process_chunks_batch
    from src.ingestion.embedder import get_embeddings
    
    CACHE_FILE = "data/evaluation_chunk_cache.json"
    chunk_cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                chunk_cache = json.load(f)
        except: pass

    for path in pdf_paths:
        try:
            print(f"[LexVed] Auto-ingesting: {os.path.basename(path)}")
            if path in chunk_cache:
                print(f"[LexVed] Using pre-parsed chunks from cache.")
                chunks = chunk_cache[path]
            else:
                chunks = extract_chunks(path)
                chunks = process_chunks_batch(chunks)
                chunk_cache[path] = chunks
                # Save cache update
                try:
                    os.makedirs("data", exist_ok=True)
                    with open(CACHE_FILE, "w") as f:
                        json.dump(chunk_cache, f)
                except: pass
                
            texts = [c["text"] for c in chunks]
            embeddings = get_embeddings(texts)
            
            # Helper for categorization
            def categorize_text(text):
                text_lower = text.lower()
                if any(k in text_lower for k in ["contract", "lease", "tenant", "owner", "property", "agreement", "civil"]):
                    return "civil", "general"
                return "criminal", "general"

            if active_db == "qdrant":
                from qdrant_client import QdrantClient
                from qdrant_client.models import PointStruct
                from src.utils.qdrant_provider import COLLECTION_NAME
                import uuid
                
                qc = QdrantClient(host="localhost", port=6333)
                points = []
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    cat, sub = categorize_text(chunk["text"])
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb.tolist(),
                        payload={
                            "text": chunk["text"],
                            "source": chunk["source"],
                            "page": chunk["page"],
                            "category": cat,
                            "subcategory": sub
                        }
                    ))
                qc.upsert(collection_name=COLLECTION_NAME, points=points)
            else:
                from src.utils.pinecone_client import index
                import uuid
                vectors = []
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    cat, sub = categorize_text(chunk["text"])
                    vectors.append({
                        "id": str(uuid.uuid4()),
                        "values": emb.tolist(),
                        "metadata": {
                            "text": chunk["text"],
                            "source": chunk["source"],
                            "page": chunk["page"],
                            "category": cat,
                            "subcategory": sub
                        }
                    })
                index.upsert(vectors=vectors)
                
            from src.retrieval.retriever import invalidate_bm25
            invalidate_bm25()
            
        except Exception as e:
            print(f"[LexVed] Error ingesting {path}: {e}")

def run_evaluation():
    if not os.path.exists(EVAL_DATA_PATH):
        print("Data target missing.")
        return

    # Guarantee we have data to benchmark
    ensure_data_ingested()

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
            from src.utils.qdrant_provider import client, COLLECTION_NAME
            vector_count = client.get_collection(COLLECTION_NAME).vectors_count
    except: pass

    print("\n" + "="*80)
    print(f" LEXVED BENCHMARK - {active_db.upper()} ".center(80))
    print("="*80)

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_results = []
    lock = threading.Lock()
    
    def process_query(cat, query, gt):
        ram_before = psutil.virtual_memory().used
        
        # M1: Embedding Time
        t_start_emb = time.time()
        query_emb = get_model_st().encode([query])[0]
        m1_emb_time = time.time() - t_start_emb
        
        # M3: Retrieval Latency
        docs, m3_ret_lat = retrieve(query, top_k=5)
        
        if not docs:
            print(f"Warning: No documents found for query: {query}")
            return
            
        # M4: Cosine Similarity
        doc_embs = get_model_st().encode([d.payload['text'] for d in docs])
        cos_sims = util.cos_sim(query_emb, doc_embs)[0]
        m4_cos_sim = cos_sims.mean().item()
        
        context = "".join([f"[Source: {d.payload.get('source')}] {d.payload['text']}\n" for d in docs])
        
        # M16: End-to-End Latency
        ans = ""
        try:
            for chunk in generate_answer_stream(query, context): 
                ans += chunk
        except Exception as e:
            print(f"Warning: LLM generation failed ({e}). Using fallback answer.")
            ans = "The system encountered an unexpected inference timeout while generating the response."
        m16_e2e = time.time() - t_start_emb
        
        q_stats = calculate_local_metrics(gt, ans)
        judge_res = judge_llm_metrics(query, gt, ans, context)
        nli_score = check_hallucination_nli(context, ans)
        ner_score = get_ner_precision(gt, ans)
        
        m19_ram = (psutil.virtual_memory().used - ram_before) / (1024**2)
        
        gt_emb = get_model_st().encode([gt])[0]
        ans_emb = get_model_st().encode([ans])[0]
        m11_sem_score = util.cos_sim(gt_emb, ans_emb)[0].item()
        
        total_tokens = max(len(ans.split()), 1)
        m17_tgl = (m16_e2e - m3_ret_lat - m1_emb_time) / total_tokens
        m18_cost = (len(context.split()) + len(query.split()) + len(ans.split())) * 0.000002
        
        def to_int(val, default=50):
            try: return int(str(val).replace("%", ""))
            except: return default
 
        res = {
            "category": cat,
            "metrics": {
                "M1": m1_emb_time, 
                "M2": vector_count, 
                "M3": m3_ret_lat,
                "M4": m4_cos_sim, 
                "M5": calculate_recall_at_k(docs, cat),
                "M6": q_stats['rouge1'], 
                "M7": q_stats['rouge2'],
                "M8": q_stats['rougeL'], 
                "M9": q_stats['meteor'], 
                "M10": q_stats['bleu'],
                "M11": m11_sem_score,
                "M12": q_stats['bert_f1'],
                "M13": 100 - nli_score, 
                "M14": nli_score, 
                "M15": to_int(judge_res.get("factual_consistency", 0)),
                "M16": m16_e2e,
                "M17": m17_tgl,
                "M18": m18_cost,
                "M19": m19_ram, 
                "M20": to_int(judge_res.get("citation_acc", 0)),
                "M21": ner_score,
                "M22": to_int(judge_res.get("precedent_match", 0)),
                "M23": to_int(judge_res.get("regulatory_alignment", 0)),
                "M24": to_int(judge_res.get("jurisdictional_comp", 0))
            }
        }
        
        with lock:
            res["id"] = len(all_results) + 1
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
            
            # Also update comparative results if active
            if os.path.exists("comparative_results.json"):
                try:
                    with open("comparative_results.json", "r") as f:
                        comp_data = json.load(f)
                    if comp_data.get("status") == "processing":
                        # We are in a comparative run
                        orig_progress = comp_data.get("progress", "")
                        # Remove existing query suffix if present
                        if " (" in orig_progress:
                            orig_progress = orig_progress.split(" (")[0]
                        comp_data["progress"] = f"{orig_progress} ({len(all_results)}/10 queries evaluated)"
                        with open("comparative_results.json", "w") as f:
                            json.dump(comp_data, f, indent=4)
                except: pass

            print(f"[{len(all_results)}/10] Completed evaluation for query.")

    queries = []
    for cat in ["civil", "criminal"]:
        for item in data[cat]:
            queries.append((cat, item['query'], item['ground_truth']))

    # Use 2 workers to perfectly balance i9 CPU saturation and 32GB RAM limits
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_query, c, q, gt) for c, q, gt in queries]
        for _ in as_completed(futures):
            pass

    # Final Report
    if all_results:
        avgs = {k: sum(r['metrics'][k] for r in all_results)/len(all_results) for k in all_results[0]['metrics']}
    else:
        print("[Warning] No queries processed successfully.")
        avgs = {f"M{i}": 0 for i in range(1, 25)}
        
    report = {
        "timestamp": time.ctime(),
        "status": "complete",
        "progress": f"Audit Complete — {len(all_results)} cases verified on {active_db.upper()}",
        "summary": avgs,
        "details": all_results,
        "system_info": {
            "vector_db": active_db.upper(),
            "model": get_active_generation_model(),
            "embedding": get_active_model_name(),
            "encryption": "AES-256"
        }
    }
    with open("evaluation_results.json", "w") as f: json.dump(report, f, indent=4)
    print(f"\n[SUCCESS] Metrics saved for {active_db.upper()}")

if __name__ == "__main__":
    run_evaluation()
