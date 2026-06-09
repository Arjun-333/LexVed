import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def test_groq_streaming_metrics():
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "Groq-Beta": "inference-metrics"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": "What is the capital of India? Answer in one word."}],
        "temperature": 0.1,
        "stream": True,
        "stream_options": {
            "include_usage": True
        }
    }
    
    t_start = time.time()
    r = requests.post(GROQ_URL, headers=headers, json=payload, stream=True, timeout=30)
    print("Status code:", r.status_code)
    
    ttft = None
    first_token_received = False
    
    for line in r.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8").strip()
        if line_str.startswith("data: "):
            data_content = line_str[6:]
            if data_content == "[DONE]":
                break
            try:
                chunk = json.loads(data_content)
                # Check for first token text
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content and not first_token_received:
                        ttft = time.time() - t_start
                        first_token_received = True
                        print(f"TTFT Measured: {ttft:.4f}s")
                
                # Check for usage/metadata
                if "x_groq" in chunk:
                    print("Found x_groq key:", chunk["x_groq"])
                if "usage" in chunk:
                    print("Found usage key:", chunk["usage"])
            except Exception as e:
                print("Error parsing chunk:", e, line_str)

if __name__ == "__main__":
    test_groq_streaming_metrics()
