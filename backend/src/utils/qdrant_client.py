from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "lexved_chunks"

def init_collection():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )