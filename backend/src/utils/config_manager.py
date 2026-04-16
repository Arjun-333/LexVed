import os
import json

CONFIG_PATH = "system_config.json"

DEFAULT_CONFIG = {
    "embedding_model": "multi-qa-mpnet-base-cos-v1",
    "models": {
        "multi-qa-MiniLM-L6-cos-v1": {"dimension": 384},
        "multi-qa-mpnet-base-cos-v1": {"dimension": 768},
        "multi-qa-distilbert-cos-v1": {"dimension": 768}
    }
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
