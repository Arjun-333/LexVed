import os
import time
from google import genai

# Read API key explicitly to allow it to be configured globally or explicitly
gemini_api_key = os.getenv("GEMINI_API_KEY")

def generate_answer(question, context, model="Gemini 1.5 Pro", max_tokens=500):
    """
    Uses Gemini 1.5 Pro to generate an answer given a document context.
    If another model is requested via the UI but you only have Gemini keys locally,
    we'll simulate it for now by telling Gemini to act like it.
    """
    prompt = f"You are {model}. Answer the question using only the context below.\n\nContext:\n{context}\n\nQ: {question}\nA:"
    
    t2 = time.time()
    
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it to your Gemini API Key.")
        
    client = genai.Client(api_key=gemini_api_key)
    
    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=max_tokens,
        )
    )
    generation_time = time.time() - t2
    
    answer = response.text
    return answer, generation_time, prompt
