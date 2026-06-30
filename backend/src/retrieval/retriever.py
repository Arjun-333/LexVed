import time
import os
import re
from src.utils.config_manager import get_active_db_name, get_active_model_params
from src.ingestion.embedder import get_embeddings
from src.utils.cache import retrieval_cache, crossencoder_cache
from rank_bm25 import BM25Okapi

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return text


global_bm25 = None
global_corpus = []

def invalidate_bm25():
    """Called when embedding model or vector DB is switched."""
    global global_bm25, global_corpus
    global_bm25 = None
    global_corpus = []
    retrieval_cache.invalidate()
    crossencoder_cache.invalidate()
    print("[LexVed] BM25 index and caches invalidated.")

def build_bm25():
    global global_bm25, global_corpus
    all_payloads = []
    
    try:
        import json
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache_path = os.path.join(backend_dir, "data", "primitive_chunk_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
            for filepath, chunks in cache_data.items():
                for c in chunks:
                    all_payloads.append({
                        "text": c.get("text", ""),
                        "source": c.get("source", ""),
                        "page": c.get("page", 1),
                        "category": c.get("category", "Uncategorized"),
                        "subcategory": c.get("subcategory", "General")
                    })
            print(f"[LexVed] BM25 built from local cache ({len(all_payloads)} chunks).")
        else:
            raise FileNotFoundError("Local chunk cache not found.")
    except Exception as local_err:
        print(f"[LexVed] Local BM25 build failed ({local_err}), falling back to database fetch...")
        active_db = get_active_db_name()
        all_payloads = []
        try:
            if active_db == "qdrant":
                from qdrant_client import QdrantClient
                from src.utils.qdrant_provider import COLLECTION_NAME
                client = QdrantClient(host="localhost", port=6333)
                offset = None
                while True:
                    res, next_offset = client.scroll(
                        collection_name=COLLECTION_NAME,
                        limit=1000,
                        offset=offset,
                        with_payload=True
                    )
                    all_payloads.extend([hit.payload for hit in res])
                    if next_offset is None:
                        break
                    offset = next_offset
            else:
                from src.utils.pinecone_client import index
                try:
                    ids = [v.id for v in index.list(limit=10000)]
                    if ids:
                        fetched = index.fetch(ids=ids)
                        all_payloads = [fetched.vectors[vid].metadata for vid in fetched.vectors]
                except Exception:
                    dim = get_active_model_params()["dimension"]
                    res = index.query(vector=[0]*dim, top_k=10000, include_metadata=True)
                    all_payloads = [match['metadata'] for match in res['matches']]
        except Exception as e:
            print(f"BM25 Build Fallback Error: {e}")

    global_corpus = all_payloads
    tokenized = [preprocess_text(p.get('text', '')).split() for p in all_payloads]
    if tokenized:
        global_bm25 = BM25Okapi(tokenized)

def get_bm25_top_k(query, top_k=20, category=None, subcategory=None):
    if global_bm25 is None:
        build_bm25()
    
    if not global_bm25:
        return []

    tokenized_query = preprocess_text(query).split()
    scores = global_bm25.get_scores(tokenized_query)
    
    scored_docs = sorted(zip(scores, global_corpus), key=lambda x: x[0], reverse=True)
    filtered = []
    for s, p in scored_docs:
        if s == 0: continue
        if category and category != "Uncategorized" and p.get("category") != category: continue
        if subcategory and subcategory != "General" and p.get("subcategory") != subcategory: continue
        filtered.append(type('Result', (), {'score': s, 'payload': p}))
        if len(filtered) >= top_k: break
    return filtered

def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    fused_scores = {}
    docs_map = {}
    
    for rank, doc in enumerate(dense_results):
        doc_id = doc.payload.get("text", "") 
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        docs_map[doc_id] = doc.payload
        
    for rank, doc in enumerate(sparse_results):
        doc_id = doc.payload.get("text", "")
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        docs_map[doc_id] = doc.payload
        
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [type('Result', (), {'score': score, 'payload': docs_map[doc_id]}) for doc_id, score in sorted_docs]

ce_model = None
def cross_encode_rerank(query, results, top_k=3):
    global ce_model
    if not results: return []
    if ce_model is None:
        from sentence_transformers import CrossEncoder
        ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Check cache for each pair, only predict uncached
    cached_scores = {}
    uncached_pairs = []
    uncached_indices = []
    for i, res in enumerate(results):
        doc_text = res.payload.get('text', '')
        cache_key = f"{query}||{doc_text[:200]}"
        cached = crossencoder_cache.get(cache_key)
        if cached is not None:
            cached_scores[i] = cached
        else:
            uncached_pairs.append([query, doc_text])
            uncached_indices.append(i)

    if uncached_pairs:
        scores = ce_model.predict(uncached_pairs)
        for j, idx in enumerate(uncached_indices):
            score = float(scores[j])
            cached_scores[idx] = score
            doc_text = results[idx].payload.get('text', '')
            cache_key = f"{query}||{doc_text[:200]}"
            crossencoder_cache.put(cache_key, score)

    for i, res in enumerate(results):
        res.score = cached_scores.get(i, 0.0)
        
    sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
    return sorted_results[:top_k]

def retrieve_qdrant(q_emb, category, subcategory, top_k):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from src.utils.qdrant_provider import COLLECTION_NAME

    # Local instantiation for absolute certainty
    client = QdrantClient(host="localhost", port=6333)

    must_filters = []
    if category and category != "Uncategorized":
        must_filters.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if subcategory and subcategory != "General":
        must_filters.append(FieldCondition(key="subcategory", match=MatchValue(value=subcategory)))

    final_filter = Filter(must=must_filters) if must_filters else None

    if hasattr(client, "search"):
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=q_emb.tolist(),
            query_filter=final_filter,
            limit=top_k,
            with_payload=True
        )
    elif hasattr(client, "query_points"):
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_emb.tolist(),
            query_filter=final_filter,
            limit=top_k,
            with_payload=True
        ).points
    else:
        # Fallback for older versions or issues
        search_result = client.search_points(
            collection_name=COLLECTION_NAME,
            query_vector=q_emb.tolist(),
            query_filter=final_filter,
            limit=top_k,
            with_payload=True
        )
    
    # Standardize result format
    results = []
    for hit in search_result:
        results.append(type('Result', (), {
            'score': hit.score,
            'payload': hit.payload
        }))
    return results

def retrieve_pinecone(q_emb, category, subcategory, top_k):
    from src.utils.pinecone_client import index
    
    filter_dict = {}
    if category and category != "Uncategorized":
        filter_dict["category"] = category
    if subcategory and subcategory != "General":
        filter_dict["subcategory"] = subcategory

    query_response = index.query(
        vector=q_emb.tolist(),
        top_k=top_k,
        filter=filter_dict if filter_dict else None,
        include_metadata=True
    )
    
    results = []
    for match in query_response['matches']:
        results.append(type('Result', (), {
            'score': match['score'],
            'payload': match['metadata']
        }))
    return results

def decompose_query(query_text: str) -> list[str]:
    """
    Decomposes a complex legal query into 2-3 simpler search-friendly queries
    to perform more targeted retrieval.
    """
    if len(query_text.strip().split()) < 6:
        return [query_text]

    import os
    import requests
    import json
    import re

    try:
        api_key = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", "")).strip()
        if not api_key:
            return [query_text]

        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        prompt = (
            "You are a legal search optimization bot.\n"
            "Analyze the legal query. If the query is complex or asks about multiple different topics, "
            "laws, or cases, split it into 2-3 distinct, simple, targeted search queries in English.\n"
            "If the query is already simple, just return it as a single-item array.\n"
            "Return ONLY a valid JSON object with a single key 'queries' containing an array of strings, "
            "with no explanation or markdown formatting.\n\n"
            f"Query: {query_text}"
        )
        payload = {
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }

        r = requests.post(url, headers=headers, json=payload, timeout=8)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
        else:
            data = json.loads(content)
        queries = data.get("queries", [query_text])
        if not isinstance(queries, list):
            queries = [query_text]
        return [str(q).strip() for q in queries if str(q).strip()][:3]
    except Exception as e:
        print(f"[LexVed] Query decomposition failed: {e}. Using original query.")
        return [query_text]

def retrieve(query_text, category=None, subcategory=None, top_k=5):
    """
    Performs retrieval using the active Vector Database (Qdrant or Pinecone).
    Implements Dynamic Query Decomposition with CrossEncoder Consolidation.
    Results are cached to avoid recomputation on repeated queries.
    """
    # Check retrieval cache first
    cache_key = f"{query_text}|{category}|{subcategory}|{top_k}|{get_active_db_name()}"
    cached = retrieval_cache.get(cache_key)
    if cached is not None:
        return cached

    t1 = time.time()
    
    # 1. Decompose the query dynamically
    sub_queries = decompose_query(query_text)
    if not sub_queries:
        sub_queries = [query_text]
        
    print(f"[LexVed] Dynamic Query Decomposition: {sub_queries}")
    
    # Collect candidates across all sub-queries
    all_candidates = []
    seen_texts = set()
    active_db = get_active_db_name()

    for sq in sub_queries:
        q_emb = get_embeddings([sq])[0]
        
        # Dense + Sparse retrieval for sub-query
        if active_db == "qdrant":
            dense_results = retrieve_qdrant(q_emb, category, subcategory, 15)
        else:
            dense_results = retrieve_pinecone(q_emb, category, subcategory, 15)
            
        sparse_results = get_bm25_top_k(sq, 15, category, subcategory)
        
        # Merge sub-query results via RRF
        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Add to consolidated pool with deduplication
        for item in fused:
            txt = item.payload.get("text", "")
            if txt not in seen_texts:
                seen_texts.add(txt)
                all_candidates.append(item)

    # 4. CrossEncoder Consolidation against ORIGINAL query
    final_results = cross_encode_rerank(query_text, all_candidates, top_k=top_k)
        
    retrieval_time = time.time() - t1
    result = (final_results, retrieval_time)
    retrieval_cache.put(cache_key, result)
    return result
