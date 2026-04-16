import time
import os
from src.utils.config_manager import get_active_db_name
from src.ingestion.embedder import get_embeddings

def retrieve_qdrant(q_emb, category, subcategory, top_k):
    from src.utils.qdrant_client import client, COLLECTION_NAME
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    must_filters = []
    if category and category != "Uncategorized":
        must_filters.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if subcategory and subcategory != "General":
        must_filters.append(FieldCondition(key="subcategory", match=MatchValue(value=subcategory)))

    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_emb.tolist(),
        query_filter=Filter(must=must_filters) if must_filters else None,
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

def retrieve(query_text, category=None, subcategory=None, top_k=5):
    """
    Performs retrieval using the active Vector Database (Qdrant or Pinecone).
    """
    q_emb = get_embeddings([query_text])[0]
    active_db = get_active_db_name()
    
    t1 = time.time()
    
    if active_db == "qdrant":
        results = retrieve_qdrant(q_emb, category, subcategory, top_k)
    else:
        results = retrieve_pinecone(q_emb, category, subcategory, top_k)
        
    retrieval_time = time.time() - t1
    
    # Page-Aware Sorting (Secondary sort handle)
    def sort_key(res):
        return (-round(res.score, 2), res.payload.get("page", 0))

    sorted_results = sorted(results, key=sort_key)
    
    return sorted_results, retrieval_time
