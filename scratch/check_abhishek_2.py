import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv("backend/.env")

def check_abhishek():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)
    
    # Query for lots of vectors
    results = index.query(vector=[0.1]*768, top_k=1000, include_metadata=True)
    
    found = False
    for i, match in enumerate(results['matches']):
        meta = match.metadata
        if "Abhishek_Banerjee" in meta.get("source", ""):
            print(f"Abhishek Result:")
            print(f"  Source: {meta.get('source')}")
            print(f"  Page: {meta.get('page')} (Type: {type(meta.get('page'))})")
            found = True
            break
    if not found:
        print("Abhishek Banerjee vectors not found in top 1000.")

if __name__ == "__main__":
    check_abhishek()
