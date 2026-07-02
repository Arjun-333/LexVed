from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return "sunny"

api_key = os.getenv("COHERE_API_KEY")
llm = ChatOpenAI(
    base_url='https://api.cohere.ai/compatibility/v1',
    api_key=api_key,
    model='command-r-plus-08-2024',
    temperature=0
)
llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke("What is the weather in Delhi?")
print("Tool calls:", getattr(response, 'tool_calls', None))
