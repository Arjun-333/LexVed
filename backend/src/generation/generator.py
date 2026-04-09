import os
import time
from google import genai

# Read API key explicitly to allow it to be configured globally or explicitly
gemini_api_key = os.getenv("GEMINI_API_KEY")

def generate_answer(question, context, model="Gemini 1.5 Pro", max_tokens=500):
    """
    Uses Gemini 1.5 Pro to generate an answer given a document context.
    Enforces strict citation of source filename and page numbers.
    """
    prompt = (
        f"You are {model}, a professional legal assistant. "
        f"Answer the question using ONLY the context provided below. "
        f"For every fact or legal point you state, you MUST cite the source and page number(s) "
        f"found in the context markers (e.g., [Source: file.pdf, Page: 4]).\n\n"
        f"Context:\n{context}\n\n"
        f"Q: {question}\nA:"
    )
    
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
