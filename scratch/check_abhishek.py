import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv("backend/.env")

def check_abhishek():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)
    
    # Query for Abhishek Banerjee
    results = index.query(vector=[0.1]*768, filter={"source": {"$contains": "Abhishek_Banerjee"}}, top_k=5, include_metadata=True)
    if not results['matches']:
        # Try a broader query
        results = index.query(vector=[0.1]*768, top_k=100, include_metadata=True)
    
    for i, match in enumerate(results['matches']):
        meta = match.metadata
        if "Abhishek_Banerjee" in meta.get("source", ""):
            print(f"Abhishek Result:")
            print(f"  Source: {meta.get('source')}")
            print(f"  Page: {meta.get('page')} (Type: {type(meta.get('page'))})")
            return

if __name__ == "__main__":
    check_abhishek()
