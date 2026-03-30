import time
from src.utils.qdrant_client import client, COLLECTION_NAME
from src.ingestion.embedder import get_embeddings

def retrieve(query, top_k=5):
    """
    Retrieves the most semantically similar chunks from Qdrant.
    """
    q_emb = get_embeddings([query])[0]
    
    t1 = time.time()
    res = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_emb.tolist(),
        limit=top_k
    )
    retrieval_time = time.time() - t1
    
    return res, retrieval_time
