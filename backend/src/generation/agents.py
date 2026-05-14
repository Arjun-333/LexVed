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
    
    # Try using Groq for ultra-fast routing
    ans = ""
    for chunk in generate_with_groq_stream(prompt, routing_model):
        ans += chunk
    
    ans = ans.strip().upper()
    
    if "COMPLEX" in ans:
        return heavy_model
    else:
        return routing_model

def execute_reasoning_agent(query: str, context: str, model: str):
    """
    The Reasoning Agent analyzes the context against the query and drafts a logical chain.
    """
    prompt = (
        "You are the LexVed Reasoning Agent. Your task is to analyze the provided legal context "
        "and draft a structured logical reasoning chain to answer the query.\n"
        "CRITICAL INSTRUCTION: You MUST rely ONLY on the provided context. Do NOT use outside knowledge. "
        "If the context does not contain sufficient information to answer the query, state explicitly: "
        "'The provided context does not contain information to answer this query.'\n"
        "Do NOT write the final answer. Just write the analytical steps, noting key facts, precedents, "
        "and any contradictions found ONLY in the text below.\n\n"
        f"Context:\n{context[:6000]}\n\n"
        f"Query: {query}\n\n"
        "Reasoning Chain:"
    )
    
    if model.startswith("llama-3"):
        return generate_with_groq_stream(prompt, model)
    else:
        return generate_with_ollama_stream(prompt, model)

def execute_synthesis_agent(query: str, reasoning_chain: str, model: str, username: str = "User"):
    """
    The Synthesis Agent takes the reasoning chain and writes the final cohesive answer.
    """
    prompt = (
        "You are the LexVed Senior Legal Synthesis Counsel. Your goal is to take the provided reasoning "
        "and craft a premium, authoritative, and human-like legal response addressed to Dear {username}. Avoid robotic structures, "
        "numbered premise lists, or 'Subject:' lines. Instead, write a cohesive narrative that flows logically.\n\n"
        "STYLE GUIDELINES:\n"
        "- Use a professional, sophisticated tone (Senior Counsel level).\n"
        "- Integrate facts and citations naturally into your prose.\n"
        "- If the reasoning indicates insufficient context, do not just give a flat refusal. "
        "Politely explain that after auditing the institutional repository, the specific factual basis "
        "required for a definitive answer was not found, and suggest where to look next.\n\n"
        "CRITICAL: DO NOT HALLUCINATE. Only use facts from the reasoning chain.\n\n"
        f"Reasoning Chain:\n{reasoning_chain}\n\n"
        f"Query: {query}\n\n"
        "Final Counsel Response:"
    )
    
    if model.startswith("llama-3"):
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
