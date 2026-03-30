import os
from src.ingestion.pdf_processor import extract_chunks, run_ner_and_redact
from src.ingestion.embedder import get_embeddings
from src.ingestion.uploader import upload_to_qdrant
from src.utils.qdrant_client import init_collection
from src.ingestion.pdf_processor import process_text

init_collection()

all_chunks = []

for root, dirs, files in os.walk("data"):
    for file in files:
        if file.endswith(".pdf"):
            path = os.path.join(root, file)

            chunks = extract_chunks(path)

            for chunk in chunks:
                chunk["text"] = process_text(chunk["text"])

            all_chunks.extend(chunks)

print(f"Total chunks: {len(all_chunks)}")

# 👉 Convert to embeddings
texts = [c["text"] for c in all_chunks]
embeddings = get_embeddings(texts)

print("✅ Embeddings created")

# 👉 Upload to Qdrant
upload_to_qdrant(all_chunks, embeddings)

print("✅ Uploaded to Qdrant")