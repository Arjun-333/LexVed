import sys
import os
import asyncio

# Ensure project backend is in path
sys.path.insert(0, '/home/arjun/Desktop/LexVed/backend')

from src.agents.workflows import stream_brief_workflow

async def main():
    print("Testing Automated Case Brief Workflow DAG...")
    query = "Balbir Kaur"
    async for event in stream_brief_workflow(query, username="test_user"):
        if event["type"] == "thought":
            print(f"\n[Thought] {event['text']}")
        elif event["type"] == "content":
            print(event["text"], end="", flush=True)
        elif event["type"] == "done":
            print(f"\n\n[Done] Generation time: {event['generation_time']:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
