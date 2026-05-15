from qdrant_client import QdrantClient
qc = QdrantClient(url="http://localhost:6333")
print(f"Has search: {hasattr(qc, 'search')}")
print(f"Has query_points: {hasattr(qc, 'query_points')}")
