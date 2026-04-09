import os
import time
import requests
import json
from google import genai
from dotenv import load_dotenv

# Load variables from .env if present
load_dotenv()

# Read API key explicitly
gemini_api_key = os.getenv("GEMINI_API_KEY")

def generate_with_gemini(prompt, model="gemini-flash-latest", max_tokens=500):
    """Generates an answer using Google Gemini API."""
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    
    client = genai.Client(api_key=gemini_api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=max_tokens,
        )
    )
    return response.text

def generate_with_ollama(prompt, model="llama3"):
    """Generates an answer using Local Ollama API (localhost:11434)."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json().get("response", "")

def generate_answer(question, context, provider="gemini", model=None, max_tokens=500):
    """
    Orchestrates generation using the requested provider (gemini or ollama).
    Enforces strict citation of source filename and page numbers.
    """
    # Default models if not specified
    if provider == "gemini":
        model = model or "gemini-flash-latest"
    else:
        model = model or "llama3"

    prompt = (
        "You are a professional legal assistant. "
        "Answer the question using ONLY the context provided below. "
        "For every fact or legal point you state, you MUST cite the source and page number(s) "
        "found in the context markers (e.g., [Source: file.pdf, Page: 4]).\n\n"
        f"Context:\n{context}\n\n"
        f"Q: {question}\nA:"
    )
    
    t2 = time.time()
    
    try:
        if provider == "gemini":
            answer = generate_with_gemini(prompt, model=model, max_tokens=max_tokens)
        elif provider == "ollama":
            answer = generate_with_ollama(prompt, model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        # Re-raise to allow app.py to handle fallback
        raise e
        
    generation_time = time.time() - t2
    return answer, generation_time, prompt
