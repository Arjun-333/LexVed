import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from src.ingestion.pdf_processor import extract_chunks, process_chunks_batch, categorize_text
from src.ingestion.embedder import get_embeddings
from src.ingestion.uploader import upload_to_qdrant
from src.utils.qdrant_client import init_collection

# Tracking for incremental ingestion
TRACKING_FILE = "ingested_files.json"

def load_tracking():
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r") as f:
            return json.load(f)
    return []

def save_tracking(ingested_list):
    with open(TRACKING_FILE, "w") as f:
        json.dump(ingested_list, f)

def process_single_pdf(path):
    """
    Worker function to process a single PDF.
    Returns the path if successful, otherwise None.
    """
    try:
        # Determine category from path
        path_upper = path.upper()
        if "CRIMINAL" in path_upper:
            dir_category = "Criminal"
        elif "CIVIL" in path_upper:
            dir_category = "Civil"
        else:
            dir_category = "Uncategorized"

        chunks = extract_chunks(path)
        if not chunks:
            return path

        # Detect subcategory from content
        sample_text = chunks[0]["text"] if chunks else ""
        _, subcategory = categorize_text(sample_text)

        # Process chunks in BATCHES (much faster)
        chunks = process_chunks_batch(chunks, batch_size=16)

        # Embed and Upload
        texts = [c["text"] for c in chunks]
        if texts:
            embeddings = get_embeddings(texts)
            upload_to_qdrant(chunks, embeddings, category=dir_category, subcategory=subcategory)
        
        return path
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        return None

def main():
    ingested_files = load_tracking()
    
    # Collect all PDF paths first
    all_pdfs = []
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)
                if path not in ingested_files:
                    all_pdfs.append(path)

    if not all_pdfs:
        print("No new documents to index.")
        return

    print(f"Scaling LexVed: Starting parallel ingestion of {len(all_pdfs)} documents...")
    
    # Use ProcessPoolExecutor to leverage i9 CPU
    # We use 8 workers to balance speed and memory (each process loads ~1GB of models)
    max_workers = 8 
    
    processed_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_pdf, path): path for path in all_pdfs}
        
        # tqdm progress bar
        with tqdm(total=len(all_pdfs), desc="Indexing PDFs", unit="pdf") as pbar:
            for future in as_completed(futures):
                result_path = future.result()
                if result_path:
                    ingested_files.append(result_path)
                    processed_count += 1
                    # Save tracking periodically
                    if processed_count % 10 == 0:
                        save_tracking(ingested_files)
                pbar.update(1)

    save_tracking(ingested_files)
    print(f"\nParallel ingestion complete. {processed_count} new documents added to Qdrant.")

if __name__ == "__main__":
    main()