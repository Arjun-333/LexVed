import time
import os
from src.utils.pinecone_client import index
from src.ingestion.embedder import get_embeddings

def retrieve(query_text, category=None, subcategory=None, top_k=5):
    """
    Performs a Semantic Search in Pinecone:
    1. Vector Similarity Search
    2. Metadata Filtering (Category/Subcategory)
    3. Page-Aware Sorting
    """
    q_emb = get_embeddings([query_text])[0]
    
    filter_dict = {}
    if category and category != "Uncategorized":
        filter_dict["category"] = category
    if subcategory and subcategory != "General":
        filter_dict["subcategory"] = subcategory

    t1 = time.time()
    
    # Pinecone Query
    query_response = index.query(
        vector=q_emb.tolist(),
        top_k=top_k * 2,
        filter=filter_dict if filter_dict else None,
        include_metadata=True
    )
    
    retrieval_time = time.time() - t1
    
    # Format results to match Qdrant-like structure for the app
    results = []
    for match in query_response['matches']:
        results.append(type('Result', (), {
            'score': match['score'],
            'payload': match['metadata']
        }))

    # Page-Aware Sorting
    def sort_key(res):
        return (-round(res.score, 2), res.payload.get("page", 0))

    sorted_results = sorted(results, key=sort_key)
    
    return sorted_results[:top_k], retrieval_time
