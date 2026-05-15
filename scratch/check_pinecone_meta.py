import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv("backend/.env")

def check_metadata():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)
    
    # Fetch some vectors
    results = index.query(vector=[0.1]*768, top_k=5, include_metadata=True)
    for i, match in enumerate(results['matches']):
        meta = match.metadata
        print(f"Result {i+1}:")
        print(f"  Source: {meta.get('source')}")
        print(f"  Page: {meta.get('page')} (Type: {type(meta.get('page'))})")
        print(f"  Text: {meta.get('text')[:100]}...")

if __name__ == "__main__":
    check_metadata()
