import sys
import os
import asyncio
import hashlib

# Ensure project backend is in path
sys.path.insert(0, '/home/arjun/Desktop/LexVed/backend')

from src.agents.graph import stream_agent
from src.agents.memory_manager import active_user

async def main():
    username = "arjun"
    active_user.set(username)
    
    message = "My name is Arjun and I am a legal counsel researching compassionate appointments."
    history = []
    
    # Calculate the exact thread_id from the screenshot
    first_q = history[0].get("question", "") if history else message
    hash_val = hashlib.md5(first_q.encode('utf-8')).hexdigest()
    thread_id = f"{username}_{hash_val}"
    
    print(f"Testing with exact thread_id: {thread_id}")
    
    async for event in stream_agent(message, history=history, thread_id=thread_id, username=username):
        print(f"EVENT: {event['type']} | {str(event.get('text', event.get('tool', '')))}")

if __name__ == "__main__":
    asyncio.run(main())
