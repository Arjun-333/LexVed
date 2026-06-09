import sys
import os
import asyncio

# Ensure project backend is in path
sys.path.insert(0, '/home/arjun/Desktop/LexVed/backend')

from src.agents.graph import stream_agent
from src.agents.memory_manager import active_user

async def main():
    active_user.set("arjun")
    
    query = "My name is Arjun and I am a legal counsel researching compassionate appointments."
    print("Testing stream_agent with greeting...")
    async for event in stream_agent(query):
        print(f"EVENT: {event['type']} | {str(event.get('text', event.get('tool', '')))}")

if __name__ == "__main__":
    asyncio.run(main())
