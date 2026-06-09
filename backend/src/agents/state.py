"""
LexVed Agent State — The Shared Memory of the Agent

HOW THIS WORKS:
    In LangGraph, "State" is a TypedDict that flows between every node in the graph.
    Each node receives the current state, does its work, and returns an UPDATED state.

    Think of it like a legal case file being passed between departments:
    - The "messages" field is the conversation transcript
    - Each tool call adds its results to the messages

    The special "Annotated[list, add_messages]" syntax tells LangGraph:
    "When a node returns new messages, APPEND them to the existing list,
     don't replace the whole list."

WHY THIS MATTERS:
    Without state, the agent would forget what tools it already called.
    With state, it can:
    1. See what the user asked
    2. See what tools it already called and their results
    3. Decide if it needs more tools or can answer now
"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The state schema for the LexVed Legal Agent.

    Fields:
        messages: The full conversation history including:
            - HumanMessage: What the user said
            - AIMessage: What the agent decided (including tool calls)
            - ToolMessage: Results from tool executions

        The add_messages annotation is a "reducer" — it tells LangGraph
        how to merge new messages into the existing list (append, not replace).
    """
    messages: Annotated[list, add_messages]
