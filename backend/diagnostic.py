import os
import sys
import torch
import time
from sentence_transformers import SentenceTransformer
import qdrant_client
import requests

def diagnostic():
    print("====================================================")
    print(" LEXVED SYSTEM DIAGNOSTIC (INTELLIGENCE COCKPIT) ")
    print("====================================================")
    print(f"Timestamp: {time.ctime()}")
    print("----------------------------------------------------")
    
    # 1. Check Embedding Model
    print("[1/5] Initializing Neural Engine (all-mpnet-base-v2)...")
    start = time.time()
    try:
        model = SentenceTransformer('all-mpnet-base-v2')
        test_emb = model.encode(["LexVed Legal Research test query"])
        elapsed = time.time() - start
        print(f"STATUS: Operational")
        print(f"LATENCY: {elapsed:.2f}s")
        print(f"DIMENSIONS: {test_emb.shape[1]}")
    except Exception as e:
        print(f"STATUS: Failed - {e}")

    # 2. Check Qdrant
    print("\n[2/5] Connecting to Vector Repository (Qdrant)...")
    try:
        client = qdrant_client.QdrantClient("localhost", port=6333)
        collections = client.get_collections()
        col_names = [c.name for c in collections.collections]
        print(f"STATUS: Connected")
        print(f"COLLECTIONS: {col_names}")
        if "lexved_chunks" in col_names:
            points = client.count("lexved_chunks").count
            print(f"INDEXED_POINTS: {points}")
    except Exception as e:
        print(f"STATUS: Connection Failed - {e}")

    # 3. Check Ollama (Llama 3)
    print("\n[3/5] Verifying Inference Node (Llama 3)...")
    try:
        # Use tags endpoint to verify model presence instead of full generation
        res = requests.get("http://localhost:11434/api/tags", timeout=5)
        if res.status_code == 200:
            models = [m['name'] for m in res.json().get('models', [])]
            print(f"STATUS: Responsive")
            print(f"AVAILABLE_MODELS: {models}")
            if any("llama3" in m for m in models):
                print("LLAMA_3: Found")
            else:
                print("LLAMA_3: Missing from local registry")
        else:
            print(f"STATUS: Unexpected response ({res.status_code})")
    except Exception as e:
        print(f"STATUS: Node Unreachable - {e}")

    # 4. Check Cache
    print("\n[4/5] Inspecting Local Intelligence Cache...")
    cache_path = os.path.expanduser("~/.cache/lexved")
    if os.path.exists(cache_path):
        size = sum(os.path.getsize(os.path.join(cache_path, f)) for f in os.listdir(cache_path) if os.path.isfile(os.path.join(cache_path, f)))
        print(f"STATUS: Active")
        print(f"LOCATION: {cache_path}")
        print(f"CACHE_SIZE: {size/1024/1024:.2f} MB")
    else:
        print("STATUS: Initializing (First run pending)")

    # 5. Check spaCy
    print("\n[5/5] Loading NLP Core (spaCy en_core_web_sm)...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("LexVed Intelligence Cockpit verification sequence.")
        print(f"STATUS: Ready")
        print(f"ENTITIES_DETECTED: {[ent.label_ for ent in doc.ents]}")
    except Exception as e:
        print(f"STATUS: Resource Missing - {e}")

    print("\n----------------------------------------------------")
    print(" DIAGNOSTIC COMPLETE ")
    print("====================================================")

if __name__ == "__main__":
    diagnostic()
