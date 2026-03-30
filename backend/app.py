from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer

app = Flask(__name__, static_folder="../frontend")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("message")
    model = data.get("model", "Gemini 1.5 Pro")  # default to our configured
    
    if not question:
        return jsonify({"error": "No message provided"}), 400
        
    print(f"User requested via {model}: {question}")
    
    try:
        # Retrieve context from Qdrant
        res, retrieval_time = retrieve(question, top_k=5)
        
        # Build context string
        context = " ".join([m.payload["text"] for m in res])
        
        # Generate Answer
        # We pass model to the generator, which can route accordingly based on keys/mock
        answer, generation_time, prompt = generate_answer(question, context, model=model, max_tokens=500)
        
        return jsonify({
            "response": answer,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
