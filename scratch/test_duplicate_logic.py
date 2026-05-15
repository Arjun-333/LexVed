import os
import sys
import hashlib
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from src.ingestion.pdf_processor import calculate_pdf_hash, extract_chunks

def test_hashing():
    # Create a dummy PDF path (even if it doesn't exist, we can mock or use a small file)
    test_file = Path(__file__).parent / "test.txt"
    test_file.write_text("This is a test legal document content.")
    
    print(f"Testing hashing for: {test_file}")
    h1 = calculate_pdf_hash(test_file)
    print(f"Hash 1: {h1}")
    
    # Change content slightly
    test_file.write_text("This is a test legal document content. Modified.")
    h2 = calculate_pdf_hash(test_file)
    print(f"Hash 2: {h2}")
    
    if h1 != h2:
        print("✓ SUCCESS: Hash changed after content modification.")
    else:
        print("✗ FAILURE: Hash did not change.")
        
    # Revert
    test_file.write_text("This is a test legal document content.")
    h3 = calculate_pdf_hash(test_file)
    if h1 == h3:
        print("✓ SUCCESS: Hash is consistent for identical content.")
    else:
        print("✗ FAILURE: Hash is inconsistent.")

if __name__ == "__main__":
    test_hashing()
