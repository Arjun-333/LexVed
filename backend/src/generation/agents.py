import os
import json
import requests
import concurrent.futures
from src.generation.generator import generate_with_groq_stream, generate_with_ollama_stream, generate_answer

def determine_query_complexity(query: str) -> str:
    """
    Uses Llama 3 8B Instant (fast model) to determine if a query is SIMPLE or COMPLEX.
    Returns the target model based on complexity.
    """
    # Fast model for routing
    routing_model = "llama-3.1-8b-instant"
    heavy_model = "llama-3.3-70b-versatile"
    
    prompt = (
        "You are an AI router. Classify the following legal query as 'SIMPLE' (direct fact retrieval, definition) "
        "or 'COMPLEX' (comparative analysis, deep reasoning, multifaceted precedent comparison).\n"
        "Return ONLY the word 'SIMPLE' or 'COMPLEX'.\n\n"
        f"Query: {query}"
    )
    
    from src.generation.generator import generate_utility
    ans = generate_utility(prompt, model=routing_model).upper()
    
    if "COMPLEX" in ans:
        return heavy_model
    else:
        return routing_model

def execute_reasoning_agent(query: str, context: str, model: str, history: list = None):
    """
    The Reasoning Agent analyzes the context and conversation history against the query.
    """
    history_str = ""
    if history:
        for m in history[-5:]: # Use last 5 messages for context
            q = m.get("question") or m.get("text", "")
            a = m.get("answer") or ""
            if q: history_str += f"User: {q}\n"
            if a: history_str += f"Assistant: {a}\n"

    prompt = (
        "You are the LexVed Reasoning Agent. Your task is to analyze the provided legal context "
        "and conversation history to draft a structured logical reasoning chain.\n"
        "STRICT INSTRUCTION: Only answer about the specific case currently being discussed in the history. "
        "If the context contains multiple 'Abhishek' cases or other irrelevant files, IGNORE THEM. "
        "Maintain absolute continuity with the established subject.\n\n"
        f"Conversation History:\n{history_str}\n"
        f"Context:\n{context[:8000]}\n\n"
        f"Query: {query}\n\n"
        "Reasoning Chain:"
    )
    
    if model.startswith("llama-3") or model.startswith("qwen-2.5") or model.startswith("mixtral") or "/" in model:
        return generate_with_groq_stream(prompt, model)
    else:
        return generate_with_ollama_stream(prompt, model)

def execute_synthesis_agent(query: str, reasoning_chain: str, model: str, username: str = "User", history: list = None):
    """
    The Synthesis Agent takes the reasoning chain and history to write the final cohesive answer.
    """
    history_str = ""
    if history:
        for m in history[-3:]:
            q = m.get("question") or m.get("text", "")
            a = m.get("answer") or ""
            if q: history_str += f"User: {q}\n"
            if a: history_str += f"Assistant: {a}\n"

    prompt = (
        f"You are the LexVed Senior Legal Synthesis Counsel. Address the user directly as {username}.\n\n"
        "MANDATORY STYLE RULES:\n"
        "- Maintain continuity with the conversation history.\n"
        "- Do NOT write like a formal letter. No 'Dear', no 'Sincerely', no 'Honorable Court'.\n"
        "- Do NOT use placeholders like '[Recipient]' or '[Your Name]'.\n"
        "- Write a high-level executive legal summary that flows naturally.\n"
        "- Every single factual claim or case mention MUST be followed by its source: [Source: filename.pdf, Page: X].\n"
        "- Integrate citations seamlessly at the end of relevant sentences.\n\n"
        "CRITICAL: NO HALLUCINATION. NO CITATION = NO MENTION.\n\n"
        f"Conversation History:\n{history_str}\n"
        f"Reasoning Chain:\n{reasoning_chain}\n\n"
        f"Query: {query}\n\n"
        "Final Counsel Response:"
    )
    
    if model.startswith("llama-3") or model.startswith("qwen-2.5") or model.startswith("mixtral") or "/" in model:
        return generate_with_groq_stream(prompt, model)
    else:
        return generate_with_ollama_stream(prompt, model)

def generate_single_model_answer(query: str, reasoning_chain: str, model: str, username: str):
    """Helper function to run a single model synchronously and return the full text."""
    ans = ""
    for chunk in execute_synthesis_agent(query, reasoning_chain, model, username):
        ans += chunk
    return {"model": model, "answer": ans}

def execute_multi_model_synthesis(query: str, reasoning_chain: str, username: str = "User"):
    """
    Runs 6 models concurrently, collects their answers, and uses an LLM Judge to pick the best one.
    """
    models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "qwen-2.5-32b",
        "mixtral-8x7b-32768",
        "qwen2.5:7b",
        "phi3"
    ]
    
    results = []
    # Run all 6 models in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(generate_single_model_answer, query, reasoning_chain, m, username) for m in models]
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if not "Error" in res["answer"]:
                    results.append(res)
            except Exception as e:
                print(f"[Multi-Model] Error generating with a model: {e}")
                
    if not results:
        yield "Error: All models failed to generate a response."
        return
        
    yield f"\n\n*Generated {len(results)} candidate answers. Evaluating to find the best...*\n\n"
    
    # Use Llama-3.3-70b as the judge
    judge_model = "llama-3.3-70b-versatile"
    
    evaluation_prompt = "You are an expert Legal AI Judge. Your task is to evaluate the following candidate answers to a legal query and pick the single best answer.\n\n"
    evaluation_prompt += f"Query: {query}\n\n"
    
    for i, res in enumerate(results):
        evaluation_prompt += f"--- Candidate {i+1} (Model: {res['model']}) ---\n"
        evaluation_prompt += res["answer"] + "\n\n"
        
    evaluation_prompt += (
        "Evaluate based on:\n"
        "1. Factual consistency with the legal domain.\n"
        "2. Lack of hallucinations.\n"
        "3. Premium, authoritative, and human-like tone.\n"
        "4. Clarity and cohesiveness.\n\n"
        "Return ONLY the exact text of the winning candidate. Do not add any commentary, introductory text, or mention which candidate won. Just output the winning response."
    )
    
    yield "*Judge has made a decision. Streaming final response...*\n\n"
    
    for chunk in generate_with_groq_stream(evaluation_prompt, judge_model):
        yield chunk
