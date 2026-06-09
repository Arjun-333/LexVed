import sys
import os
import asyncio

# Ensure project backend is in path
sys.path.insert(0, '/home/arjun/Desktop/LexVed/backend')

from src.agents.graph import run_agent, get_agent

async def main():
    print("Testing synchronous agent run...")
    try:
        res = run_agent("Does the introduction of a Family Benefit Scheme completely extinguish a dependent's right to claim compassionate appointment?")
        print("Final Answer:")
        print(res["answer"])
        print("Tools used:", res["tool_calls"])
    except Exception as e:
        print("Error during sync run:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
