import os
import json
import random
import time
import requests
import re
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def generate_query_and_gt(chunk_text):
    prompt = f"""You are an expert legal scholar. Given the following legal document chunk, generate:
1. A realistic, complex legal query/question that can be answered ONLY using this chunk.
2. A detailed, professional ground-truth answer based strictly on the chunk.

Legal Chunk:
{chunk_text[:3000]}

Return ONLY a JSON object:
{{
  "query": "realistic query here",
  "ground_truth": "detailed ground truth answer here"
}}"""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(5):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                m = re.search(r'\{.*\}', content, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
                    if parsed.get("query") and parsed.get("ground_truth"):
                        return parsed
            elif r.status_code == 429:
                time.sleep(15)
            else:
                time.sleep(3)
        except Exception:
            time.sleep(3)
    return None

def main():
    print("[*] Loading chunks from primitive cache...")
    cache_path = "data/primitive_chunk_cache.json"
    if not os.path.exists(cache_path):
        print("[ERROR] primitive_chunk_cache.json not found.")
        return

    with open(cache_path, "r") as f:
        cache_data = json.load(f)

    # Flatten cache to a list of chunks
    all_chunks = []
    for filepath, chunks in cache_data.items():
        for c in chunks:
            text = c.get("text", "").strip()
            if len(text) > 300:
                all_chunks.append(text)

    print(f"[*] Total chunks available: {len(all_chunks)}")
    
    # Randomly sample 100 chunks
    random.seed(42)
    sampled_chunks = random.sample(all_chunks, min(100, len(all_chunks)))
    
    dataset = []
    print(f"[*] Generating 100 synthetic queries...")
    for idx, chunk in enumerate(sampled_chunks):
        print(f"Generating query {idx+1}/100...")
        res = generate_query_and_gt(chunk)
        if res:
            dataset.append({
                "query": res["query"],
                "ground_truth": res["ground_truth"],
                "gold_chunk_text": chunk
            })
            # Sleep slightly to stay within limits
            time.sleep(0.5)
        else:
            print(f"[Warning] Failed to generate for chunk {idx+1}")

    out_path = "data/synthetic_evaluation_dataset.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"[SUCCESS] Generated {len(dataset)} synthetic queries at '{out_path}'.")

if __name__ == "__main__":
    main()
