import os
import json
import contextvars

# Thread-safe context variable to track the active logged-in user
active_user = contextvars.ContextVar("active_user", default="unknown")
retrieval_counter = contextvars.ContextVar("retrieval_counter", default=0)

MEMORY_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "long_term_memory.json"
)

def _load_memory() -> dict:
    """Load the long term memory file securely, creating it if not present."""
    if not os.path.exists(os.path.dirname(MEMORY_FILE_PATH)):
        os.makedirs(os.path.dirname(MEMORY_FILE_PATH), exist_ok=True)
        
    if not os.path.exists(MEMORY_FILE_PATH):
        with open(MEMORY_FILE_PATH, "w") as f:
            json.dump({}, f)
        return {}
        
    try:
        with open(MEMORY_FILE_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[LexVed Memory] Error reading memory file: {e}")
        return {}

def _save_memory(memory_data: dict):
    """Save the long term memory file securely."""
    try:
        with open(MEMORY_FILE_PATH, "w") as f:
            json.dump(memory_data, f, indent=2)
    except Exception as e:
        print(f"[LexVed Memory] Error saving memory file: {e}")

def store_user_fact(key: str, value: str):
    """Store a structured fact or preference for the active user context."""
    username = active_user.get()
    memory = _load_memory()
    
    if username not in memory:
        memory[username] = {}
        
    memory[username][key] = value
    _save_memory(memory)
    print(f"[LexVed Memory] Stored fact for user '{username}': {key} -> {value}")

def retrieve_user_facts() -> dict:
    """Retrieve all stored facts/preferences for the active user context."""
    username = active_user.get()
    memory = _load_memory()
    return memory.get(username, {})
