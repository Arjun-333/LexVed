from sentence_transformers import SentenceTransformer
from src.utils.config_manager import get_active_model_name
import numpy as np

_model = None
_current_model_name = None
_cohere_client = None

def get_model():
    global _model, _current_model_name
    active_name = get_active_model_name()
    if _model is None or _current_model_name != active_name:
        # Cohere models require the Cohere SDK, not SentenceTransformer
        if "Cohere" in active_name or "cohere" in active_name:
            print(f"[LexVed] Loading Cohere Embedding Model: {active_name}")
            _model = "cohere"
        else:
            print(f"[LexVed] Loading Embedding Model: {active_name}")
            _model = SentenceTransformer(active_name)
        _current_model_name = active_name
    return _model

def _get_cohere_client():
    global _cohere_client
    if _cohere_client is None:
        import os
        try:
            import cohere
            api_key = os.getenv("COHERE_API_KEY", "")
            _cohere_client = cohere.ClientV2(api_key)
        except ImportError:
            raise ImportError("Cohere SDK not installed. Run: pip install cohere")
    return _cohere_client

def get_embeddings(texts):
    model = get_model()
    if model == "cohere":
        client = _get_cohere_client()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.embed(
                    texts=list(texts),
                    model="embed-english-v3.0",
                    input_type="search_document",
                    embedding_types=["float"]
                )
                return np.array(response.embeddings.float_)
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    wait = 2 ** attempt
                    print(f"[LexVed] Cohere embed failed (attempt {attempt+1}), retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    print(f"[LexVed] Cohere embed failed after {max_retries} attempts: {e}")
                    raise
    return model.encode(texts, convert_to_numpy=True)