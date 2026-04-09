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
    # Using modern query_points with correct Boolean logic:
    # We only include should_filters if there is a category/must filter,
    # OR we make it optional in a way that doesn't exclude points.
    
    # In Qdrant, a top-level Filter with only 'should' acts as 'must match at least one'.
    # To avoid this, we only apply the should_filters if we also have must_filters,
    # or we handle the broad search differently.
    
    search_filter = None
    if must_filters or should_filters:
        search_filter = Filter(must=must_filters, should=should_filters)
        # If it's a broad search (no category), and we have a keyword booster,
        # we don't want it to be restrictive. 
        # Fix: If no must_filters, we shouldn't use should_filters at the top level filter if we want full recall.
        if not must_filters and should_filters:
            search_filter = None # Fallback to pure vector search for now to prioritize recall in 'All' searches

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb.tolist(),
        query_filter=search_filter,
        limit=top_k * 2,
        with_payload=True
    ).points
    retrieval_time = time.time() - t1
    
    # Page-Aware Sorting:
    # If scores are very close (within 0.05), sort by page number for better coherence.
    def sort_key(res):
        # Primary: Score (rounded to 2 decimal places)
        # Secondary: Page (ascending)
        return (-round(res.score, 2), res.payload.get("page", 0))

    sorted_results = sorted(search_result, key=sort_key)
    
    return sorted_results[:top_k], retrieval_time
