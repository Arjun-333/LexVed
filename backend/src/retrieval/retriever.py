import time
import os
from src.utils.qdrant_client import client, COLLECTION_NAME
from src.ingestion.embedder import get_embeddings
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText

def retrieve(query_text, category=None, subcategory=None, top_k=5):
    """
    Performs a Hybrid Search in Qdrant:
    1. Vector Similarity Search
    2. Keyword Matching (using Should filters)
    3. Page-Aware Sorting for results with similar scores
    """
    q_emb = get_embeddings([query_text])[0]
    
    must_filters = []
    if category:
        must_filters.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if subcategory:
        must_filters.append(FieldCondition(key="subcategory", match=MatchValue(value=subcategory)))

    # Hybrid Search: Add a 'should' match for exact keywords
    should_filters = [
        FieldCondition(key="text", match=MatchText(text=query_text))
    ]

    t1 = time.time()
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_emb.tolist(),
        query_filter=Filter(
            must=must_filters,
            should=should_filters
        ),
        limit=top_k * 2, # Fetch more to allow for re-ranking/sorting
        with_payload=True
    )
    retrieval_time = time.time() - t1
    
    # Page-Aware Sorting:
    # If scores are very close (within 0.05), sort by page number for better coherence.
    def sort_key(res):
        # Primary: Score (rounded to 2 decimal places)
        # Secondary: Page (ascending)
        return (-round(res.score, 2), res.payload.get("page", 0))

    sorted_results = sorted(search_result, key=sort_key)
    
    return sorted_results[:top_k], retrieval_time
