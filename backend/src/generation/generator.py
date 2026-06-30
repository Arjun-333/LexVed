import os
import time
import requests
import json
from dotenv import load_dotenv
from src.utils.config_manager import get_active_generation_model, load_config

load_dotenv()

def generate_with_ollama_stream(prompt, model=None):
    """Yields chunks of the generated answer from Local Ollama API."""
    model = model or get_active_generation_model()

    if model in load_config().get("groq_models", []):
        yield from generate_with_groq_stream(prompt, model)
        return

    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not ollama_host:
        ollama_host = "http://127.0.0.1:11434"
    if ollama_host.startswith(":"):
        ollama_host = f"http://127.0.0.1{ollama_host}"
    elif not ollama_host.startswith("http://") and not ollama_host.startswith("https://"):
        ollama_host = f"http://{ollama_host}"
    url = f"{ollama_host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096 # Limit context window to keep reasoning fast
        }
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with requests.post(url, json=payload, stream=True, timeout=120) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        yield chunk.get("response", "")
                        if chunk.get("done"):
                            break
                return  # Success — exit retry loop
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                import time
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"[LexVed] Ollama connection failed (attempt {attempt+1}/{max_retries}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                yield f"\n\n[Error: Could not connect to Ollama at {url} after {max_retries} attempts. Is the service running?]"
        except requests.exceptions.HTTPError as e:
            yield f"\n\n[Error: Ollama returned HTTP {e.response.status_code}. Model '{model}' may not be available.]"
            return

def generate_with_groq_stream(prompt, model):
    url = "https://router.huggingface.co/v1/chat/completions"
    api_key = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", "")).strip()
    if not api_key:
        yield "Error: HF_TOKEN not found in environment."
        return

    # Map model if it's a legacy Groq name
    model_mapping = {
        "llama-3.1-8b-instant": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3.3-70b-versatile": "meta-llama/Llama-3.3-70B-Instruct",
        "mixtral-8x7b-32768": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "qwen-2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    }
    mapped_model = model_mapping.get(model, model)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": mapped_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.2
    }
    max_retries = 6
    for attempt in range(max_retries):
        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=12) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: ") and line_str != "data: [DONE]":
                            try:
                                chunk = json.loads(line_str[6:])
                                yield chunk["choices"][0]["delta"].get("content", "")
                            except:
                                pass
                return  # Success
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)
            else:
                yield f"\n\n[Error: Hugging Face API unreachable after {max_retries} attempts.]"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    import time
                    print(f"[LexVed] Hugging Face rate limit hit during generation. Waiting 30s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(30)
                    continue
                else:
                    yield f"\n\n[Error: Hugging Face rate limit exhausted after {max_retries} attempts.]"
                    return
            else:
                yield f"\n\n[Error: Hugging Face returned HTTP {e.response.status_code}.]"
            return

def generate_utility(prompt: str, model: str = "meta-llama/Llama-3.1-8B-Instruct") -> str:
    """Fast, non-streaming generation for internal logic (routing, condensation)."""
    ans = ""
    # Try Hugging Face first for speed
    try:
        for chunk in generate_with_groq_stream(prompt, model):
            if "[Rate limit reached" in chunk: break # Stop if fallback triggered
            ans += chunk
        
        if ans.strip(): return ans.strip()
    except:
        pass
        
    # Fallback to Local Ollama
    ans = ""
    for chunk in generate_with_ollama_stream(prompt, model="llama3:8b"):
        ans += chunk
    return ans.strip()

def generate_answer_stream(question, context, model=None, history=None):
    """
    Generator that provides a legal answer and its metadata.
    Trims context to prevent extreme latency.
    Supports multi-turn conversation via optional history.
    """
    model = model or get_active_generation_model()
    
    # Trim context to ~6000 chars to ensure fast inference on local hardware
    trimmed_context = context[:6000] 

    # Build conversation history prefix
    history_prefix = ""
    if history and len(history) > 0:
        history_prefix = "Previous conversation:\n"
        for turn in history[-3:]:  # Last 3 turns
            history_prefix += f"Q: {turn.get('question', '')}\nA: {turn.get('answer', '')}\n\n"
        history_prefix += "Now answer the following new question based on the context and previous conversation.\n\n"

    prompt = (
        "You are the LexVed Universal Intelligence Agent. "
        "Answer the question primarily using the provided legal context. "
        "CRITICAL: If the context is insufficient, you may supplement with your internal pre-trained legal expertise, "
        "but you MUST clearly state when you are using general knowledge versus context-specific data.\n\n"
        "CITATIONS: For context-specific data, you MUST cite the exact [Source: filename.pdf, Page: X] "
        "found in the context headers. Do NOT use legal citations like 'AIR 1989' for the [Source] tag; "
        "use the literal filename provided in the context.\n\n"
        f"{history_prefix}"
        f"Context:\n{trimmed_context}\n\n"
        f"Q: {question}\nA:"
    )
    return generate_with_ollama_stream(prompt, model=model)

def generate_answer(question, context, model=None, history=None):
    """Non-streaming version of generate_answer for legacy endpoints."""
    ans = ""
    for chunk in generate_answer_stream(question, context, model=model, history=history):
        ans += chunk
    return ans, 0, 0 # Return 0 for times as they are handled elsewhere or ignored
