from sentence_transformers import SentenceTransformer

model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")

def get_embeddings(texts):
    return model.encode(texts, convert_to_numpy=True)