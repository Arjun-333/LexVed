import sys
import os

# Add backend to path
sys.path.append(os.getcwd())

from src.utils.qdrant_provider import init_collection, client, COLLECTION_NAME
from src.ingestion.uploader import upload_to_qdrant
from src.retrieval.retriever import retrieve
import numpy as np

def test_subindexing():
    print("Starting Sub-indexing Verification...")
    
    # 1. Reset collection
    init_collection()
    print("Collection initialized.")
    
    # 2. Prepare dummy data
    chunks = [
        {"text": "Robbery is a criminal offense."},
        {"text": "Financial fraud involves deception."},
        {"text": "Property disputes are civil matters."},
        {"text": "Marriage laws vary by state."}
    ]
    
    # Create random embeddings (size 768 to match config)
    embeddings = np.random.rand(4, 768)
    
    # 3. Upload with categories
    print("Uploading categorized data...")
    upload_to_qdrant([chunks[0]], [embeddings[0]], category="Criminal", subcategory="Robbery")
    upload_to_qdrant([chunks[1]], [embeddings[1]], category="Criminal", subcategory="Fraud")
    upload_to_qdrant([chunks[2]], [embeddings[2]], category="Civil", subcategory="Property")
    upload_to_qdrant([chunks[3]], [embeddings[3]], category="Civil", subcategory="Marriage")
    
    # 4. Test filtering
    print("\n🔍 Testing Retrieval with Filters:")
    
    # Test case 1: Criminal filter
    res, _ = retrieve("criminal", category="Criminal")
    print(f"  Target: Criminal | Hits: {len(res)}")
    for r in res:
        print(f"    - [{r.payload['category']}/{r.payload['subcategory']}] {r.payload['text']}")
    assert all(r.payload['category'] == "Criminal" for r in res)
    
    # Test case 2: Criminal -> Robbery filter
    res, _ = retrieve("robbery", category="Criminal", subcategory="Robbery")
    print(f"  Target: Criminal/Robbery | Hits: {len(res)}")
    for r in res:
        print(f"    - [{r.payload['category']}/{r.payload['subcategory']}] {r.payload['text']}")
    assert all(r.payload['category'] == "Criminal" and r.payload['subcategory'] == "Robbery" for r in res)
    
    # Test case 3: Unfiltered
    res, _ = retrieve("laws")
    print(f"  Target: All | Hits: {len(res)}")
    assert len(res) == 4
    
    print("\nSub-indexing verified successfully!")

if __name__ == "__main__":
    try:
        test_subindexing()
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        sys.exit(1)
