import os, sys, time, json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))
load_dotenv()

from src.ingestion.pdf_processor import extract_chunks, process_chunks_batch
from src.ingestion.embedder import get_embeddings
from src.utils.pinecone_client import index
from src.ingestion.uploader import upload_to_pinecone

def reingest():
    pdf_dir = BACKEND_DIR / "data" / "PDF"
    pdf_files = list(pdf_dir.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs. Starting Turbo Ingestion (Optimized Batches)...")

    # 1. Parallel Extraction (I/O Bound)
    all_chunks = []
    print("Step 1: Extracting text from PDFs (Parallel I/O)...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(lambda p: extract_chunks(str(p)), pdf_files), total=len(pdf_files), desc="Extraction"))
        for r in results: all_chunks.extend(r)
    
    print(f"Extracted {len(all_chunks)} raw chunks.")

    # 2. Batch Processing (SpaCy NER + Categorization)
    print("Step 2: Processing chunks (NER + Redaction + Categorization)...")
    # process_chunks_batch uses nlp.pipe internally which is already optimized
    processed_chunks = process_chunks_batch(all_chunks, batch_size=128)
    
    # 3. Batch Embedding
    print("Step 3: Embedding chunks (Torch Batching)...")
    texts = [c["text"] for c in processed_chunks]
    t0 = time.time()
    # get_embeddings uses batch_size=32 internally
    embeddings = get_embeddings(texts)
    print(f"Embedding done in {time.time()-t0:.1f}s")
    
    # 4. Parallel Upsert to Pinecone
    print("Step 4: Upserting to Pinecone (Parallel Batches)...")
    batch_size = 100
    def upsert_worker(start_idx):
        end_idx = min(start_idx + batch_size, len(processed_chunks))
        batch_chunks = processed_chunks[start_idx:end_idx]
        batch_embeddings = embeddings[start_idx:end_idx]
        upload_to_pinecone(batch_chunks, batch_embeddings)
    
    upsert_ranges = list(range(0, len(processed_chunks), batch_size))
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(upsert_worker, upsert_ranges), total=len(upsert_ranges), desc="Upserting"))

    print(f"Successfully re-ingested {len(processed_chunks)} chunks into Pinecone (legal-hybrid-rag).")

if __name__ == "__main__":
    reingest()
