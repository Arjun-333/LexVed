import os
import time
import requests
import json
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

def generate_with_ollama_stream(prompt, model=None):
    """Yields chunks of the generated answer from Local Ollama API."""
    model = model or OLLAMA_MODEL
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096 # Limit context window to keep reasoning fast
        }
    }
    
    with requests.post(url, json=payload, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("response", "")
                if chunk.get("done"):
                    break

def generate_answer_stream(question, context, model=None):
    """
    Generator that provides a legal answer and its metadata.
    Trims context to prevent extreme latency.
    """
    model = model or OLLAMA_MODEL
    
    # Trim context to ~3000 chars to ensure fast inference on local hardware
    trimmed_context = context[:6000] 

    prompt = (
        "You are a professional legal assistant. "
        "Answer the question using ONLY the provided context. "
        "Cite the source and page number(s) (e.g., [Source: file.pdf, Page: 4]).\n\n"
        f"Context:\n{trimmed_context}\n\n"
        f"Q: {question}\nA:"
    )
    
    return generate_with_ollama_stream(prompt, model=model)
def generate_answer(question, context, model=None):
    """Non-streaming version of generate_answer for legacy endpoints."""
    ans = ""
    for chunk in generate_answer_stream(question, context, model=model):
        ans += chunk
    return ans, 0, 0 # Return 0 for times as they are handled elsewhere or ignored
