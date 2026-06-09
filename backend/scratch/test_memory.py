import sys
import os
import asyncio

# Ensure project backend is in path
sys.path.insert(0, '/home/arjun/Desktop/LexVed/backend')

from src.agents.graph import run_agent
from src.agents.memory_manager import active_user

def test_long_term_memory():
    print("--- Testing Long-Term Memory ---")
    active_user.set("arjun")
    
    # Store a fact
    from src.agents.tools import remember_legal_fact, recall_legal_facts
    print("Storing fact...")
    remember_legal_fact.invoke({"fact_key": "client_name", "fact_value": "Balbir Kaur"})
    remember_legal_fact.invoke({"fact_key": "opposing_counsel", "fact_value": "Senior Advocate Mr. Sharma"})
    
    # Recall
    print("Recalling facts...")
    res = recall_legal_facts.invoke({})
    print(res)
    assert "Balbir Kaur" in res
    assert "opposing_counsel" in res
    print("[SUCCESS] Long-term memory works perfectly!")

async def test_short_term_memory():
    print("\n--- Testing Short-Term Memory Checkpointing ---")
    thread_id = "test_thread_123"
    username = "arjun"
    
    # Turn 1: Introduce ourselves
    print("Sending message 1...")
    res1 = run_agent("Hello! My name is Arjun and I am researching the Balbir Kaur case.", thread_id=thread_id, username=username)
    print(f"Agent response: {res1['answer']}")
    
    # Turn 2: Ask what my name is (relies on short-term memory checkpointer!)
    print("\nSending message 2 (relies on memory)...")
    res2 = run_agent("What is my name and which case am I researching?", thread_id=thread_id, username=username)
    print(f"Agent response: {res2['answer']}")
    
    # Assert
    assert "Arjun" in res2["answer"] or "arjun" in res2["answer"].lower()
    print("[SUCCESS] Short-term checkpointing works perfectly!")

if __name__ == "__main__":
    test_long_term_memory()
    asyncio.run(test_short_term_memory())
