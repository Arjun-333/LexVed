import os
from src.ingestion.pdf_processor import extract_chunks, run_ner_and_redact
from src.ingestion.embedder import get_embeddings
from src.ingestion.uploader import upload_to_qdrant
from src.utils.qdrant_client import init_collection
from src.ingestion.pdf_processor import process_text, categorize_text

init_collection()

all_data = [] # List of (chunks, embeddings, category, subcategory)

for root, dirs, files in os.walk("backend/data"):
    for file in files:
        if file.endswith(".pdf"):
            path = os.path.join(root, file)
            print(f"Processing {path}...")

            # More robust category detection: Check if CRIMINAL or CIVIL is in the path
            path_upper = path.upper()
            if "CRIMINAL" in path_upper:
                dir_category = "Criminal"
            elif "CIVIL" in path_upper:
                dir_category = "Civil"
            else:
                dir_category = "Uncategorized"

            chunks = extract_chunks(path)
            
            # Detect subcategory from content
            sample_text = chunks[0]["text"] if chunks else ""
            _, subcategory = categorize_text(sample_text)
            
            print(f"  Final Tag: {dir_category} -> {subcategory}")

            for chunk in chunks:
                chunk["text"] = process_text(chunk["text"])

            texts = [c["text"] for c in chunks]
            if texts:
                embeddings = get_embeddings(texts)
                upload_to_qdrant(chunks, embeddings, category=dir_category, subcategory=subcategory)

print("All documents processed and uploaded to Qdrant with sub-indexing.")