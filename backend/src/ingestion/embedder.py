from sentence_transformers import SentenceTransformer
from src.utils.config_manager import get_active_model_name, get_active_db_name
import numpy as np
import os
import time

_model = None
_current_model_name = None
_cohere_client = None

def get_model():
    global _model, _current_model_name
    active_name = get_active_model_name()
    if _model is None or _current_model_name != active_name:
        # Cohere models require the Cohere SDK, not SentenceTransformer
        if "Cohere" in active_name or "cohere" in active_name or "embed-english" in active_name:
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

class CohereEmbedder:
    def __init__(self, api_key=None):
        import cohere
        self.api_key = api_key or os.getenv("COHERE_API_KEY", "")
        self.client = cohere.ClientV2(self.api_key)
        self.model = "embed-english-v3.0"
        print(f"[LexVed] CohereEmbedder initialized with model: {self.model}")

    def encode(self, texts, batch_size=96, show_progress_bar=True, input_type="search_document"):
        all_emb = []
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Cohere encoding")
        for i in iterator:
            batch = texts[i:i + batch_size]
            resp = self.client.embed(
                texts=list(batch),
                model=self.model,
                input_type=input_type,
                embedding_types=["float"]
            )
            all_emb.extend(resp.embeddings.float_)
        return np.array(all_emb)

    def encode_query(self, texts):
        resp = self.client.embed(
            texts=list(texts),
            model=self.model,
            input_type="search_query",
            embedding_types=["float"]
        )
        return np.array(resp.embeddings.float_)