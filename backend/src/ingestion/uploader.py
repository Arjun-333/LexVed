from src.utils.pinecone_client import index
import uuid

def upload_to_pinecone(chunks, embeddings, category="Uncategorized", subcategory="General"):
    """
    Upserts vector embeddings and metadata to Pinecone.
    """
    vectors = []
    for i, emb in enumerate(embeddings):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": emb.tolist(),
            "metadata": {
                **chunks[i],
                "category": chunks[i].get("category", category),
                "subcategory": chunks[i].get("subcategory", subcategory)
            }
        })

    # Upsert in batches of 100
    for i in range(0, len(vectors), 100):
        batch = vectors[i:i+100]
        index.upsert(vectors=batch)

def upload_to_qdrant(chunks, embeddings, category="Uncategorized", subcategory="General"):
    """
    Upserts vector embeddings and metadata to Qdrant.
    """
    from src.utils.qdrant_provider import client, COLLECTION_NAME
    from qdrant_client.models import PointStruct
    
    points = []
    for i, emb in enumerate(embeddings):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={
                **chunks[i],
                "category": chunks[i].get("category", category),
                "subcategory": chunks[i].get("subcategory", subcategory)
            }
        ))

    # Upload points in batches
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )