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
                "category": category,
                "subcategory": subcategory
            }
        })

    # Upsert in batches of 100
    for i in range(0, len(vectors), 100):
        batch = vectors[i:i+100]
        index.upsert(vectors=batch)