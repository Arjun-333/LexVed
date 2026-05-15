import requests
import json
import os
import sys
from pathlib import Path

# Config
API_URL = "http://127.0.0.1:5000"
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.append(str(BACKEND_DIR))

from src.utils.auth import create_token

def test_full_system():
    print("\n" + "="*50)
    print("  LEXVED DEVELOPER BRANCH - MASTER TEST")
    print("="*50)

    # 1. Generate Auth Token
    token = create_token("admin", "admin")
    headers = {"Authorization": f"Bearer {token}"}
    print(f"✓ Auth Token Generated")

    # 2. Find a test PDF
    test_pdf = next(BACKEND_DIR.rglob("*.pdf"))
    print(f"✓ Using Test PDF: {test_pdf.name}")

    # 3. Test Ingestion (First Time)
    print(f"--- Step 1: Initial Ingestion ---")
    with open(test_pdf, "rb") as f:
        files = {"file": (test_pdf.name, f, "application/pdf")}
        r = requests.post(f"{API_URL}/api/ingest", headers=headers, files=files, stream=True)
        
        found_processing = False
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                print(f"  [Stream] {data.get('step')}")
                if "Extracted" in data.get('step', ''):
                    found_processing = True
        
    if found_processing:
        print("✓ SUCCESS: Initial ingestion triggered processing.")
    else:
        print("⚠ NOTE: File might already be indexed. Testing duplicate skip next.")

    # 4. Test Duplicate Skip (Second Time)
    print(f"\n--- Step 2: Duplicate Ingestion Check ---")
    with open(test_pdf, "rb") as f:
        files = {"file": (test_pdf.name, f, "application/pdf")}
        r = requests.post(f"{API_URL}/api/ingest", headers=headers, files=files, stream=True)
        
        found_skip = False
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                print(f"  [Stream] {data.get('step')}")
                if "Skipping redundant ingestion" in data.get('step', ''):
                    found_skip = True
        
    if found_skip:
        print("✓ SUCCESS: System correctly identified and skipped duplicate document.")
    else:
        print("✗ FAILURE: System failed to detect duplicate document.")

    # 5. Test Deep Linking Metadata
    print(f"\n--- Step 3: Deep Link Metadata Verification ---")
    chat_payload = {
        "message": f"Summary of {test_pdf.name}?",
        "agentic": False
    }
    r = requests.post(f"{API_URL}/api/chat", headers=headers, json=chat_payload, stream=True)
    
    found_sources = False
    for line in r.iter_lines():
        if line:
            data = json.loads(line)
            if data.get("type") == "metadata":
                sources = data.get("sources", [])
                if sources:
                    found_sources = True
                    print(f"  [Metadata] Sources found: {len(sources)}")
                    s = sources[0]
                    print(f"  [Metadata] First Source Path: {s.get('path')}")
                    if s.get("path"):
                         print("✓ SUCCESS: Citation metadata contains full file path for deep linking.")
                    else:
                         print("✗ FAILURE: Citation metadata is missing file path.")
    
    if not found_sources:
        print("⚠ NOTE: No sources retrieved for this query. Deep link test partial.")

    print("\n" + "="*50)
    print("  TESTING COMPLETE")
    print("="*50)

if __name__ == "__main__":
    test_full_system()
