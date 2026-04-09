from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "lexved_chunks"

def init_collection():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    
    # Create payload indices for fast filtering (Sub-indexing)
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="category",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="subcategory",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="page",
        field_schema=PayloadSchemaType.INTEGER,
    )
    # Enable Full-Text Search for Hybrid Search logic
    from qdrant_client.models import TextIndexParams, TokenizerType
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="text",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.WORD,
            lowercase=True,
            replace_whitespace=True
        ),
    )