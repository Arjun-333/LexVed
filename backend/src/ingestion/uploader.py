from src.utils.qdrant_client import client, COLLECTION_NAME
from qdrant_client.models import PointStruct
import uuid

def upload_to_qdrant(chunks, embeddings, category="Uncategorized", subcategory="General"):
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={
                **chunks[i],
                "category": category,
                "subcategory": subcategory
            }
        )
        for i, emb in enumerate(embeddings)
    ]

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=100
    )