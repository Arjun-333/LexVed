import os
import sys
from pathlib import Path

# Add backend to sys.path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.generation.generator import generate_with_groq_stream

def test_fallback():
    prompt = "Say 'Fallback Success' if you are reading this from local Ollama."
    # Use a non-existent model name to force a potential error or just trigger fallback manually
    # But wait, I'll trigger 429 by... well I can't easily.
    
    print("Starting stream...")
    # I'll manually trigger the fallback logic in a temporary version of generator.py if needed.
    # Actually, I'll just check if generate_with_ollama_stream works first.
    from src.generation.generator import generate_with_ollama_stream
    for chunk in generate_with_ollama_stream("Hi", model="llama3:8b"):
        print(chunk, end="", flush=True)
    print("\nOllama check done.")

if __name__ == "__main__":
    test_fallback()
