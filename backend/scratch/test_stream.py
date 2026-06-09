import sys
import os
import asyncio

# Ensure project backend is in path
sys.path.insert(0, '/home/arjun/Desktop/LexVed/backend')

from src.agents.graph import stream_agent

async def main():
    print("Testing streaming agent...")
    query = "Does the introduction of a Family Benefit Scheme completely extinguish a dependent's right to claim compassionate appointment?"
    
    async for event in stream_agent(query):
        print(f"EVENT: {event['type']} | {str(event.get('text', event.get('tool', '')))}")

if __name__ == "__main__":
    asyncio.run(main())
