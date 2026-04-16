from sentence_transformers import SentenceTransformer
from src.utils.config_manager import get_active_model_name

_model = None
_current_model_name = None

def get_model():
    global _model, _current_model_name
    active_name = get_active_model_name()
    if _model is None or _current_model_name != active_name:
        print(f"[LexVed] Loading Embedding Model: {active_name}")
        _model = SentenceTransformer(active_name)
        _current_model_name = active_name
    return _model

def get_embeddings(texts):
    return get_model().encode(texts, convert_to_numpy=True)