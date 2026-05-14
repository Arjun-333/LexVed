import os, sys, torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Add backend to sys.path to import from primitive_pipeline_v2
sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()

from sentence_transformers import SentenceTransformer
import cohere
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, AutoModel
import requests

def test_model(name, loader_fn):
    print(f"\n--- Testing {name} ---")
    try:
        model = loader_fn()
        test_text = ["This is a test legal document."]
        if hasattr(model, "encode"):
            emb = model.encode(test_text)
            if isinstance(emb, dict):
                # BGE-M3 returns a dict
                shape = emb["dense_vecs"].shape if "dense_vecs" in emb else "unknown"
            else:
                shape = emb.shape
            print(f"✅ {name} Success! Embedding shape: {shape}")
        elif hasattr(model, "embed"): # Cohere
            resp = model.embed(texts=test_text, model="embed-english-v3.0", input_type="search_document")
            print(f"✅ {name} Success! Embedding shape: {np.array(resp.embeddings).shape}")
        else:
            print(f"⚠ {name} loaded but no encode method found.")
    except Exception as e:
        print(f"❌ {name} Failed: {e}")

def load_mpnet(): return SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-cos-v1")
def load_minilm(): return SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
def load_distilbert(): return SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1")

def load_cohere():
    key = os.getenv("COHERE_API_KEY", "[REMOVED_COHERE_KEY]")
    return cohere.Client(api_key=key)

def test_groq():
    print("\n--- Testing Groq ---")
    key = os.getenv("GROQ_API_KEY", "[REMOVED_GROQ_KEY]")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": "Say hello"}], "temperature": 0.1}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code == 200:
            print(f"✅ Groq Success! Response: {r.json()['choices'][0]['message']['content']}")
        else:
            print(f"❌ Groq Failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"❌ Groq Failed: {e}")

def load_bge_m3():
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

def load_e5_mistral():
    print("Checking E5-Mistral (Skipping full load due to 7B size on CPU, but checking config)...")
    # Just check if we can load tokenizer/config to verify ID
    AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")
    return "Tokenizer Loaded (Logic OK)"

if __name__ == "__main__":
    test_model("MPNet", load_mpnet)
    test_model("MiniLM", load_minilm)
    test_model("DistilBERT", load_distilbert)
    test_model("BGE-M3", load_bge_m3)
    test_model("Cohere", load_cohere)
    test_groq()
    # test_model("E5-Mistral", load_e5_mistral)
    print("\n--- Model Verification Complete ---")
