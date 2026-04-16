import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

from src.utils.config_manager import get_active_model_params

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "lexved-index")

# Create index if it doesn't exist
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    params = get_active_model_params()
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

index = pc.Index(INDEX_NAME)
