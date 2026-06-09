import sys
import os
import asyncio

# Ensure project backend is in path
sys.path.insert(0, '/home/arjun/Desktop/LexVed/backend')

from src.agents.graph import get_agent

async def main():
    agent = get_agent()
    query = "Does the introduction of a Family Benefit Scheme completely extinguish a dependent's right to claim compassionate appointment?"
    
    async for event in agent.astream_events({"messages": [("user", query)]}, version="v2"):
        kind = event["event"]
        name = event["name"]
        print(f"EVENT: {kind} | NAME: {name}")
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            print(f"  -> CHUNK: {repr(chunk.content)} | TOOL_CALLS: {getattr(chunk, 'tool_call_chunks', None)}")

if __name__ == "__main__":
    asyncio.run(main())
