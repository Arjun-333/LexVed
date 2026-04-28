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
    import time
    from pinecone import ServerlessSpec
    
    current_indexes = pc.list_indexes()
    index_names = [idx.name for idx in current_indexes]
    
    if INDEX_NAME in index_names:
        # Find index details
        idx_desc = next((idx for idx in current_indexes if idx.name == INDEX_NAME), None)
        
        if idx_desc and idx_desc.dimension == params["dimension"]:
            print(f"[LexVed] Pinecone index '{INDEX_NAME}' dimension matches ({params['dimension']}).")
            print("[LexVed] Wiping vectors instead of deleting index to prevent Conflict 409 errors...")
            try:
                # Use delete_all=True for default namespace
                index_inst = pc.Index(INDEX_NAME)
                index_inst.delete(delete_all=True)
                return
            except Exception as e:
                print(f"[LexVed] delete_all failed ({e}). Proceeding to delete/recreate.")
                
        # Dimension mismatch or wipe failed — delete index
        print(f"[LexVed] Deleting Pinecone index '{INDEX_NAME}' for clean slate...")
        pc.delete_index(INDEX_NAME)
        
        # Poll until deleted (max 2 mins)
        for _ in range(24):
            time.sleep(5)
            if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
                break
            print("[LexVed] Waiting for index deletion...")
            
    # Create the index
    print(f"[LexVed] Creating Pinecone index '{INDEX_NAME}' (Dim: {params['dimension']})...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=params["dimension"],
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
    # Wait for index to be ready
    for _ in range(24):
        desc = pc.describe_index(INDEX_NAME)
        if desc.status['ready']:
            break
        print("[LexVed] Waiting for index to be ready...")
        time.sleep(5)

# Auto-init if missing on import
current_idx_names = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in current_idx_names:
    create_index()

index = pc.Index(INDEX_NAME)
