import requests
import json

url = "http://127.0.0.1:11434/api/generate"
payload = {
    "model": "llama3:8b",
    "prompt": "Say hello",
    "stream": False
}

try:
    response = requests.post(url, json=payload, timeout=60)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json().get('response')}")
except Exception as e:
    print(f"Error: {e}")
