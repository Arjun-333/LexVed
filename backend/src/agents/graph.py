"""
LexVed Agent Graph — The Brain That Decides What To Do

HOW LANGGRAPH WORKS (Step by Step):

    LangGraph is a state machine. It has:
    - NODES: Functions that process data (like stations on a railway)
    - EDGES: Connections between nodes (like the railway tracks)
    - STATE: The data that travels along the tracks (like the train)

    The flow is:

        START → agent_node → (should I use a tool?) 
                    ↓ YES                    ↓ NO
              tool_node              END (return answer)
                    ↓
              agent_node (loop back with tool results)

    This creates a LOOP:
    1. Agent decides → "I need retrieve_documents"
    2. Tool runs → returns legal docs
    3. Agent decides again → "I also need extract_citations" 
    4. Tool runs → returns citations
    5. Agent decides again → "I have everything, here's my answer"
    6. END

WHAT EACH PIECE DOES:

    1. ChatGroq: The LLM brain (Groq's Llama 3.3 70B, same as your existing setup)
    2. bind_tools(): Tells the LLM "here are the tools you can call"
    3. agent_node(): Sends the conversation to the LLM, gets back a decision
    4. tool_node(): If the LLM decided to call a tool, this runs it
    5. should_continue(): Checks if the LLM wants more tools or is done

THE KEY INSIGHT:
    The LLM doesn't "run" tools itself. It returns a structured message saying
    "I want to call retrieve_documents with query='...'". Then LangGraph
    intercepts that message and actually runs the function. The result
    gets sent back to the LLM as a ToolMessage.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.agents.state import AgentState
from src.agents.tools import ALL_TOOLS


# ─── Step 1: Create the LLM with Tool Awareness ─────────────────
#
# ChatGroq connects to Groq's API (same key you already have in .env)
# bind_tools() attaches our tool definitions so the LLM knows what's available
#
# When the LLM processes a message, it can now respond in TWO ways:
#   a) Normal text response (no tools needed)
#   b) A "tool_call" response: {"name": "retrieve_documents", "args": {"query": "..."}}

def _create_llm():
    """Create the Groq LLM with tools bound.
    
    Uses the GROQ_API_KEY from your .env file.
    Model is configurable via system_config.json — but for the agent brain,
    we use a strong model (llama-3.3-70b) because it needs to REASON
    about which tools to use. Smaller models make poor tool decisions.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,          # Deterministic tool selection (no randomness)
        api_key=os.getenv("GROQ_API_KEY"),
    )
    # bind_tools tells the LLM: "You have these functions available"
    # The LLM reads each tool's name, description, and parameter types
    return llm.bind_tools(ALL_TOOLS)


# ─── Step 2: The Agent Node ─────────────────────────────────────
#
# This is the "thinking" node. It receives the full conversation history
# and decides what to do next.
#
# The system prompt is critical — it tells the LLM HOW to behave.
# Without it, the LLM might try to answer legal questions from memory
# instead of using the retrieval tool.

SYSTEM_PROMPT = """You are LexVed, an expert Indian legal research assistant powered by a comprehensive knowledge base of Supreme Court and High Court case laws.

CRITICAL RULES:
1. For ANY legal question, you MUST use the retrieve_documents tool first. Never answer legal questions from memory.
2. After retrieving documents, you MAY use extract_citations to identify specific legal references.
3. Only use extract_entities when the user specifically asks about parties, judges, or locations in a case.
4. Only use deidentify_text when the user explicitly asks to anonymize or redact text.
5. For simple greetings or non-legal questions, respond directly without tools.

RESPONSE STYLE:
- Be precise and professional
- Always cite the source documents when answering legal questions
- If the retrieved documents don't contain relevant information, say so honestly
- Structure complex answers with clear sections"""


def agent_node(state: AgentState) -> dict:
    """
    The Agent's Brain — processes messages and decides the next action.

    How it works:
        1. Takes the current conversation (state["messages"])
        2. Prepends the system prompt (so the LLM knows it's a legal assistant)
        3. Sends everything to Groq's LLM
        4. The LLM returns either:
           - A text response (done, no tools needed)
           - A tool_call (needs to use a tool before answering)
        5. We return the response as a new message in the state

    Args:
        state: Current AgentState with message history

    Returns:
        Updated state with the agent's response message appended
    """
    from langchain_core.messages import SystemMessage

    llm = _create_llm()

    # Prepend system prompt to give the LLM its identity and rules
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # The LLM processes all messages and returns its decision
    response = llm.invoke(messages)

    # Return the response — LangGraph appends it to state["messages"]
    # via the add_messages reducer we defined in state.py
    return {"messages": [response]}


# ─── Step 3: The Routing Logic ──────────────────────────────────
#
# After the agent node runs, we need to decide:
#   - Did the LLM request a tool call? → Route to tool_node
#   - Did the LLM give a final answer? → Route to END
#
# We check this by looking at the last message's "tool_calls" field.
# If it has tool calls, the LLM wants to use tools.
# If it doesn't, the LLM is done thinking.

def should_continue(state: AgentState) -> str:
    """
    The Router — decides if the agent needs more tools or is finished.

    This function is called after every agent_node execution.
    It looks at the last message from the LLM:
    - If the message contains tool_calls → return "tools" (go to tool_node)
    - If the message is plain text → return "end" (we're done)

    Returns:
        "tools" to route to the tool executor, or "end" to finish
    """
    last_message = state["messages"][-1]

    # Check if the LLM's response contains tool calls
    # tool_calls is a list like: [{"name": "retrieve_documents", "args": {...}}]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


# ─── Step 4: Build the Graph ────────────────────────────────────
#
# This is where we wire everything together into a state machine.
#
# The graph looks like:
#
#     START
#       ↓
#     agent  ←──────┐
#       ↓            │
#   (should_continue?)│
#       ↓ "tools"    │
#     tools ─────────┘
#       ↓ "end"
#      END
#
# ToolNode is a prebuilt LangGraph component that:
# 1. Reads the tool_calls from the agent's last message
# 2. Finds the matching tool function from ALL_TOOLS
# 3. Calls it with the arguments the LLM specified
# 4. Returns the result as a ToolMessage

def build_agent_graph():
    """
    Constructs the LangGraph agent graph with an active MemorySaver checkpointer.

    Architecture:
        1. StateGraph(AgentState) — Creates a graph that uses our state schema
        2. add_node("agent", ...) — Registers the agent brain
        3. add_node("tools", ...) — Registers the tool executor
        4. set_entry_point("agent") — Start at the agent node
        5. add_conditional_edges — After agent, check should_continue
        6. add_edge("tools", "agent") — After tools, always go back to agent
        7. compile(checkpointer=MemorySaver()) — Locks the graph into an executable stateful machine

    Returns:
        A compiled LangGraph agent ready to process messages
    """
    from langgraph.checkpoint.memory import MemorySaver

    # Create the graph with our state schema
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("agent", agent_node)          # The LLM brain
    graph.add_node("tools", ToolNode(ALL_TOOLS))  # The tool executor

    # Set where to start
    graph.set_entry_point("agent")

    # After agent runs, check if we need tools or are done
    graph.add_conditional_edges(
        "agent",                # After this node...
        should_continue,        # ...run this function to decide where to go
        {
            "tools": "tools",   # If should_continue returns "tools" → go to tools node
            "end": END,         # If should_continue returns "end" → finish
        }
    )

    # After tools run, ALWAYS go back to agent (so it can process the results)
    graph.add_edge("tools", "agent")

    # Compile with memory checkpointing
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ─── Step 5: The Public Interface ────────────────────────────────
#
# This is what app.py will call. It takes a user message,
# runs it through the entire agent graph, and returns the final answer.
#
# The streaming version yields chunks as the agent thinks and acts,
# so the frontend can show real-time progress.

# Singleton graph instance (compiled once, reused for all requests)
_agent_graph = None


def get_agent():
    """Get or create the singleton agent graph instance."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


def run_agent(user_message: str, history: list = None, thread_id: str = "default_thread", username: str = "unknown") -> dict:
    """
    Run the LexVed agent with a user message.

    This is the main entry point for non-streaming usage.

    Args:
        user_message: The user's question or command
        history: Optional list of previous {"question": ..., "answer": ...} dicts
        thread_id: Unique identifier to retrieve conversation checkpoints
        username: The active username to track memory context

    Returns:
        dict with:
            - "answer": The final text response
            - "tool_calls": List of tools that were used
            - "steps": Number of reasoning steps taken
    """
    from langchain_core.messages import HumanMessage, AIMessage
    from src.agents.memory_manager import active_user
    
    # Securely set user context for thread safety
    active_user.set(username)

    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}

    # Retrieve existing state if checkpointer has history for this thread
    state = agent.get_state(config)
    messages = []

    # If the thread has no history yet, seed it with the provided history list
    if not state.values or not state.values.get("messages"):
        if history:
            for turn in history:
                messages.append(HumanMessage(content=turn["question"]))
                messages.append(AIMessage(content=turn["answer"]))
    
    # Add the current user query
    messages.append(HumanMessage(content=user_message))

    # Run the graph — state checkpoint handles merging and memory
    result = agent.invoke({"messages": messages}, config=config)

    # Extract the final answer (last AI message without tool calls)
    final_messages = result["messages"]
    final_answer = ""
    tool_calls_made = []

    for msg in final_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append(tc["name"])
        if hasattr(msg, "content") and not getattr(msg, "tool_calls", None):
            if msg.type == "ai":
                final_answer = msg.content

    return {
        "answer": final_answer,
        "tool_calls": tool_calls_made,
        "steps": len(final_messages),
    }


async def stream_agent(user_message: str, history: list = None, thread_id: str = "default_thread", username: str = "unknown"):
    """
    Stream the LexVed agent's execution for real-time frontend updates.

    Yields JSON-serializable dicts for each event:
        - {"type": "thought", "text": "..."} — Agent is thinking
        - {"type": "tool_call", "tool": "...", "args": {...}} — Tool being called
        - {"type": "tool_result", "tool": "...", "result": "..."} — Tool finished
        - {"type": "content", "text": "..."} — Final answer chunk
        - {"type": "done", "tool_calls": [...], "steps": N} — Complete

    Args:
        user_message: The user's question
        history: Optional conversation history
        thread_id: Unique thread identifier for state checkpoint persistence
        username: The active username to track memory context
    """
    from langchain_core.messages import HumanMessage, AIMessage
    from src.agents.memory_manager import active_user

    # Securely set user context for thread safety
    active_user.set(username)

    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}

    # Retrieve existing state from the checkpoint
    state = agent.get_state(config)
    messages = []

    if not state.values or not state.values.get("messages"):
        if history:
            for turn in history:
                messages.append(HumanMessage(content=turn["question"]))
                messages.append(AIMessage(content=turn["answer"]))
    
    messages.append(HumanMessage(content=user_message))

    # Stream events from the graph
    tool_calls_made = []
    final_answer = ""

    async for chunk in agent.astream({"messages": messages}, config=config):
        if "agent" in chunk:
            ai_msg = chunk["agent"]["messages"][-1]
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                for tc in ai_msg.tool_calls:
                    tool_calls_made.append(tc["name"])
                    yield {
                        "type": "tool_call",
                        "tool": tc["name"],
                        "args": tc.get("args", {})
                    }
            elif hasattr(ai_msg, "content") and ai_msg.content:
                # Final answer reached!
                final_answer = ai_msg.content
                
                # Stream final answer word-by-word to the frontend for premium visual pacing
                import asyncio
                words = final_answer.split(" ")
                for i, word in enumerate(words):
                    # Keep spacing natural
                    spacing = "" if i == len(words) - 1 else " "
                    yield {"type": "content", "text": word + spacing}
                    await asyncio.sleep(0.015)

        elif "tools" in chunk:
            tool_msg = chunk["tools"]["messages"][-1]
            yield {
                "type": "tool_result",
                "tool": tool_msg.name,
                "result": str(tool_msg.content)[:500] + ("..." if len(str(tool_msg.content)) > 500 else ""),
            }

    yield {
        "type": "done",
        "tool_calls": tool_calls_made,
        "steps": len(tool_calls_made) + 1,
    }
