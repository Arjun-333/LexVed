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
        "Cohere/Cohere-embed-english-v3.0": {"dimension": 1024}
    },
    "generation_models": ["llama3", "llama3:70b", "qwen2.5:70b", "qwen2.5:7b", "mistral", "phi3", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
    "groq_models": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama-3.1-8b-instant"],
    "providers": ["qdrant", "pinecone"]
}

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
