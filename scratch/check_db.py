import os
import sys
from qdrant_client import QdrantClient
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv("backend/.env")

def check_status():
    print("--- Qdrant Check ---")
    try:
        qc = QdrantClient(host="localhost", port=6333)
        cols = qc.get_collections().collections
        print(f"Collections: {[c.name for c in cols]}")
        for c in cols:
            count = qc.count(c.name).count
            print(f"  - {c.name}: {count} points")
    except Exception as e:
        print(f"Qdrant Error: {e}")

    print("\n--- Pinecone Check ---")
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"Index: {index_name}")
        print(f"Total Vectors: {stats.total_vector_count}")
        print(f"Namespaces: {stats.namespaces}")
    except Exception as e:
        print(f"Pinecone Error: {e}")

if __name__ == "__main__":
    check_status()
