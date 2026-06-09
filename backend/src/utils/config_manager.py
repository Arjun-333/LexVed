import os
import json

CONFIG_PATH = "system_config.json"

DEFAULT_CONFIG = {
    "generation_model": "llama3",
    "embedding_model": "multi-qa-mpnet-base-cos-v1",
    "vector_db": "qdrant",
    "models": {
        "multi-qa-MiniLM-L6-cos-v1": {"dimension": 384},
        "multi-qa-mpnet-base-cos-v1": {"dimension": 768},
        "multi-qa-distilbert-cos-v1": {"dimension": 768},
        "BAAI/bge-m3": {"dimension": 1024},
        "intfloat/multilingual-e5-large-instruct": {"dimension": 1024},
        "embed-english-v3.0": {"dimension": 1024}
    },
    "generation_models": ["ensemble", "llama3", "llama-3.1-8b-instant", "llama3:70b", "qwen2.5:70b", "qwen2.5:7b", "mistral", "phi3", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
    "groq_models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "qwen-2.5-32b"],
    "providers": ["qdrant", "pinecone"]
}


def _derive_model_display_name(model_id):
    """Derive a human-readable display name from a model ID string."""
    display_map = {
        "ensemble": "Ensemble Logic",
        "llama3": "Local Llama 3 8B",
        "llama-3.1-8b-instant": "Llama 3.1 8B",
        "llama3:70b": "Llama 3 70B",
        "llama-3.3-70b-versatile": "Llama 3.3 70B",
        "qwen2.5:7b": "Local Qwen 2.5",
        "qwen2.5:70b": "Qwen 2.5 70B",
        "mistral": "Mistral 7B",
        "phi3": "Phi-3",
        "mixtral-8x7b-32768": "Mixtral 8x7B",
        "qwen-2.5-32b": "Qwen 2.5 32B",
    }
    return display_map.get(model_id, model_id)


def _derive_model_tier(model_id):
    """Derive the tier (cloud/local) from the model ID."""
    config = load_config()
    groq_models = config.get("groq_models", [])
    if model_id == "ensemble":
        return "orchestrator"
    if model_id in groq_models:
        return "cloud"
    return "local"


def _derive_model_icon(model_id):
    """Derive a Material Icons icon name from the model ID."""
    if model_id == "ensemble":
        return "hub"
    if "70b" in model_id:
        return "auto_awesome"
    if "mixtral" in model_id or "8x7b" in model_id:
        return "all_inclusive"
    if "instant" in model_id or "mistral" in model_id:
        return "bolt"
    if "qwen" in model_id:
        return "security"
    if "phi" in model_id:
        return "science"
    return "memory"


def get_generation_model_metadata():
    """Returns metadata for all generation models — used by frontend to render UI dynamically."""
    config = load_config()
    gen_models = config.get("generation_models", [])
    return [
        {
            "id": m,
            "name": _derive_model_display_name(m),
            "icon": _derive_model_icon(m),
            "tier": _derive_model_tier(m),
        }
        for m in gen_models
    ]

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except:
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

def get_active_model_name():
    return load_config()["embedding_model"]

def get_active_model_params():
    config = load_config()
    model_name = config["embedding_model"]
    return config["models"].get(model_name, {"dimension": 768})

def set_active_model(model_name):
    config = load_config()
    if model_name in config["models"]:
        config["embedding_model"] = model_name
        save_config(config)
        return True
    return False

def get_active_db_name():
    return load_config().get("vector_db", "qdrant")

def set_active_db(db_name):
    config = load_config()
    if db_name in config.get("providers", ["qdrant", "pinecone"]):
        config["vector_db"] = db_name
        save_config(config)
        return True
    return False

def get_active_generation_model():
    return load_config().get("generation_model", "llama3")

def set_active_generation_model(model_name):
    config = load_config()
    if model_name in config.get("generation_models", []):
        config["generation_model"] = model_name
        save_config(config)
        return True
    return False
