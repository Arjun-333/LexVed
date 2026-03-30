import os
from src.ingestion.pdf_processor import extract_chunks, run_ner_and_redact

all_chunks = []

for root, dirs, files in os.walk("data"):
    for file in files:
        if file.endswith(".pdf"):
            path = os.path.join(root, file)

            chunks = extract_chunks(path)

            for chunk in chunks:
                from src.ingestion.pdf_processor import process_text

                chunk["text"] = process_text(chunk["text"])

            all_chunks.extend(chunks)

print(f"Total chunks: {len(all_chunks)}")

print("\nSample chunk:\n", all_chunks[0]["text"])