from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer
from src.ingestion.pdf_processor import categorize_text

def detect_category_from_query(query):
    """
    Uses the same categorization logic to detect category/subcategory from user query.
    In a production system, this could be a separate LLM call.
    """
    return categorize_text(query)

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
    model = data.get("model", "Gemini 1.5 Pro")
    
    if not question:
        return jsonify({"error": "No message provided"}), 400
        
    print(f"User requested via {model}: {question}")
    
    try:
        # Detect category and subcategory from question
        category, subcategory = detect_category_from_query(question)
        print(f"Detected Category: {category}, Subcategory: {subcategory}")

        # Retrieve context from Qdrant with filters
        res, retrieval_time = retrieve(
            question, 
            top_k=5, 
            category=category if category != "Uncategorized" else None,
            subcategory=subcategory if subcategory != "General" else None
        )
        
        # Build context string with metadata markers for LLM citation
        context = ""
        for m in res:
            source = os.path.basename(m.payload.get("source", "Unknown"))
            page = m.payload.get("page", "?")
            context += f"\n[Source: {source}, Page: {page}]\n{m.payload['text']}\n"
        
        # If no context found with filters, fallback to unfiltered search
        if not context.strip():
            print("No context found with filters, falling back to all-index search.")
            res, _ = retrieve(question, top_k=5)
            for m in res:
                source = os.path.basename(m.payload.get("source", "Unknown"))
                page = m.payload.get("page", "?")
                context += f"\n[Source: {source}, Page: {page}]\n{m.payload['text']}\n"

        # Generate Answer
        answer, generation_time, prompt = generate_answer(question, context, model=model, max_tokens=500)
        
        return jsonify({
            "response": answer,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "category": category,
            "subcategory": subcategory
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
