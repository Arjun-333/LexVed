import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

from src.utils.config_manager import get_active_model_params

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "lexved-index")

def create_index():
    params = get_active_model_params()
    if INDEX_NAME in [idx.name for idx in pc.list_indexes()]:
        pc.delete_index(INDEX_NAME)
    
    from pinecone import ServerlessSpec
    pc.create_index(
        name=INDEX_NAME,
        dimension=params["dimension"],
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Auto-init if missing on import
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    create_index()

index = pc.Index(INDEX_NAME)
