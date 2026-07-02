import sys
import os
import asyncio

# Ensure project backend is in path
sys.path.insert(0, '/home/arjun/Desktop/LexVed/backend')

from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer_stream
from src.agents.graph import stream_agent

async def test_standard_mode(query: str):
    print("\n" + "="*80)
    print(f"Executing Query in STANDARD Mode: '{query}'")
    print("="*80)
    
    # 1. Retrieval
    res, retrieval_time = retrieve(query, top_k=5)
    print(f"[Retrieval] Retrieved {len(res)} chunks in {retrieval_time:.2f}s")
    
    context = ""
    for m in res:
        source = os.path.basename(m.payload.get("source", "Unknown"))
        page_val = m.payload.get("page")
        display_page = page_val if (isinstance(page_val, int) and page_val > 0) else 1
        context += f"\n[Source: {source}, Page: {display_page}]\n{m.payload['text']}\n"
        
    # 2. Generation Stream
    print("[Generation] Generating answer stream:")
    for chunk in generate_answer_stream(query, context, model="llama-3.1-8b-instant"):
        print(chunk, end="", flush=True)
    print("\n")

async def test_agentic_mode(query: str):
    print("\n" + "="*80)
    print(f"Executing Query in AGENTIC Mode (LangGraph): '{query}'")
    print("="*80)
    
    # Run the stream_agent generator
    async for event in stream_agent(query, thread_id="test_compare_thread", username="test_user"):
        if event["type"] == "thought":
            print(f"\n[Thought] {event['text']}")
        elif event["type"] == "tool_call":
            print(f"\n[Tool Call] {event['tool']} with args: {event.get('args', {})}")
        elif event["type"] == "tool_result":
            print(f"\n[Tool Result] Completed {event['tool']}.")
        elif event["type"] == "content":
            print(event["text"], end="", flush=True)
        elif event["type"] == "done":
            print(f"\n\n[Done] Completed with {event.get('steps', 0)} steps.")
    print("\n")

async def main():
    query = "liability under Section 138 of NI Act"
    
    # Run standard mode
    await test_standard_mode(query)
    
    # Run agentic mode
    await test_agentic_mode(query)

if __name__ == "__main__":
    asyncio.run(main())
