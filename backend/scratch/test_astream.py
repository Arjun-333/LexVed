import sys
import os
import asyncio

# Ensure project backend is in path
sys.path.insert(0, '/home/arjun/Desktop/LexVed/backend')

from src.agents.graph import get_agent
from src.agents.memory_manager import active_user

async def main():
    active_user.set("arjun")
    agent = get_agent()
    config = {"configurable": {"thread_id": "test_astream_thread"}}
    
    query = "Does the introduction of a Family Benefit Scheme completely extinguish a dependent's right to claim compassionate appointment?"
    
    print("Streaming with astream...")
    async for chunk in agent.astream({"messages": [("user", query)]}, config=config):
        print(f"\nCHUNK KEY: {list(chunk.keys())}")
        for node, val in chunk.items():
            print(f"  NODE: {node}")
            if "messages" in val:
                last_msg = val["messages"][-1]
                print(f"  LAST MSG TYPE: {type(last_msg).__name__}")
                print(f"  LAST MSG CONTENT PREVIEW: {str(last_msg.content)[:200]}")
                if hasattr(last_msg, "tool_calls"):
                    print(f"  TOOL CALLS: {last_msg.tool_calls}")
            else:
                print(f"  VAL keys: {list(val.keys())}")

if __name__ == "__main__":
    asyncio.run(main())
