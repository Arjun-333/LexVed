import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

from src.utils.config_manager import get_active_model_name, get_active_model_params

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

def get_index_name():
    params = get_active_model_params()
    dim = params["dimension"]
    return f"lexved-audit-{dim}"

def get_namespace():
    model_name = get_active_model_name()
    mapping = {
        "multi-qa-MiniLM-L6-cos-v1": "multi-qa-MiniLM-L6-cos-v1",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": "multi-qa-MiniLM-L6-cos-v1",
        "multi-qa-mpnet-base-cos-v1": "multi-qa-mpnet-base-cos-v1",
        "sentence-transformers/multi-qa-mpnet-base-cos-v1": "multi-qa-mpnet-base-cos-v1",
        "multi-qa-distilbert-cos-v1": "multi-qa-distilbert-cos-v1",
        "sentence-transformers/multi-qa-distilbert-cos-v1": "multi-qa-distilbert-cos-v1",
        "BAAI/bge-m3": "BAAI/bge-m3",
        "bge-m3": "BAAI/bge-m3",
    }
    if model_name in mapping:
        return mapping[model_name]
    return model_name.split('/')[-1].lower().replace('_', '-').replace('.', '-')

INDEX_NAME = get_index_name()
NAMESPACE = get_namespace()

def get_index_instance():
    name = get_index_name()
    current_idx_names = []
    import time
    for attempt in range(5):
        try:
            current_idx_names = [idx.name for idx in pc.list_indexes()]
            break
        except Exception as e:
            print(f"[LexVed] Network error fetching indexes, retrying in 5s (Attempt {attempt+1}/5): {e}")
            time.sleep(5)
            
    if name not in current_idx_names:
        create_index(name)
    return pc.Index(name)

def create_index(name=None):
    if name is None:
        name = get_index_name()
    params = get_active_model_params()
    from pinecone import ServerlessSpec
    print(f"[LexVed] Creating pooled index '{name}' for Dim: {params['dimension']}...")
    
    # Check if exists
    current_indexes = [idx.name for idx in pc.list_indexes()]
    if name in current_indexes:
        print(f"[LexVed] Index {name} already exists.")
        return

    pc.create_index(
        name=name,
        dimension=params["dimension"],
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    # Wait for ready
    import time
    for _ in range(24):
        desc = pc.describe_index(name)
        if desc.status['ready']: break
        time.sleep(5)

# Provide a proxy for 'index' that automatically injects the namespace
class IndexProxy:
    def __getattr__(self, attr):
        # Return a version of the index that always uses the current model's namespace
        idx = get_index_instance()
        ns = get_namespace()
        
        # If the attribute is a method that takes a namespace, wrap it
        original_attr = getattr(idx, attr)
        if callable(original_attr):
            def wrapper(*args, **kwargs):
                if 'namespace' not in kwargs:
                    kwargs['namespace'] = ns
                return original_attr(*args, **kwargs)
            return wrapper
        return original_attr
    
    def __getitem__(self, key):
        return get_index_instance()[key]

index = IndexProxy()
