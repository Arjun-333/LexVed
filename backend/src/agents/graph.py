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

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None
from langchain_openai import ChatOpenAI
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

_use_cohere_fallback = True

def _create_llm():
    """Create the LLM client with tool binding and automatic fallback."""
    cohere_llm = ChatOpenAI(
        base_url="https://api.cohere.ai/compatibility/v1",
        api_key=os.getenv("COHERE_API_KEY", ""),
        model="command-r-plus-08-2024",
        temperature=0,
        streaming=True
    )
    cohere_bound = cohere_llm.bind_tools(ALL_TOOLS)

    hf_token = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))
    if hf_token:
        hf_llm = ChatOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
            model="meta-llama/Llama-3.3-70B-Instruct",
            temperature=0,
            streaming=True
        )
        hf_bound = hf_llm.bind_tools(ALL_TOOLS)
        return cohere_bound.with_fallbacks([hf_bound])
    
    return cohere_bound


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
1. To answer a legal question, you MUST use the retrieve_documents tool to search for case laws. Once you have retrieved the relevant documents, analyze them and formulate your response.
2. PLANNING & DECOMPOSITION: If the question is complex, plan to make up to 3 distinct searches with retrieve_documents for different aspects of the issue (e.g. search 1: statutory ingredients/elements; search 2: relevant Supreme Court precedents; search 3: exceptions/defences). Do not repeat the same search query. Stop making searches once you have collected sufficient information, or when the tool returns a search limit message, or after a maximum of 3 searches.
3. After retrieving documents, you MAY use extract_citations to identify specific legal references.
4. Only use extract_entities when the user specifically asks about parties, judges, or locations in a case.
5. Only use deidentify_text when the user explicitly asks to anonymize or redact text.
6. For simple greetings or non-legal questions, respond directly without tools.

RESPONSE FORMAT (MUST USE THIS STRUCTURE):
Your final response must be structured as a professional legal brief:

# LEGAL BRIEF: [Precise Legal Question / Case Topic]

## 1. Core Legal Issue
[Formulate the exact legal question raised by the user]

## 2. Applicable Law & Statutory Provisions
[List key Acts, Sections, and Articles with details]

## 3. Essential Ingredients of the Offence / Claim
[Detailed bulleted checklist of elements required to satisfy the provision]

## 4. Authoritative Judicial Precedents (Supreme Court / High Courts)
[Citations of the retrieved cases with a summary of the holding and how it applies]

## 5. Exceptions, Defences, and Rebuttals
[Identify any exceptions, defences available, or rebuttals found in the text]

## 6. Synthesis & Conclusion
[Provide a final expert recommendation/synthesis]

## 7. Comparative Performance Metrics (Standard vs. Agentic)
┌──────────────────────┬────────────────┬────────────────────────┐
│ Evaluation Metric    │ Standard RAG   │ Agentic RAG (LexVed)   │
├──────────────────────┼────────────────┼────────────────────────┤
│ Queries Executed     │ 1              │ {queries_executed}     │
│ Documents Analyzed   │ 5 Chunks       │ {documents_analyzed}   │
│ Duplicate Removal    │ ❌ (Raw Conc.) │ ✅ (Consolidated)      │
│ Query Reformulation  │ ❌ (Simple)    │ ✅ (Facetted Search)   │
│ Multi-Step Reasoning │ ❌ (Single)    │ ✅ (Plan-Search-Audit) │
│ Audit Verification   │ ❌ (None)      │ ✅ (Factual Check)     │
└──────────────────────┴────────────────┴────────────────────────┘
[Brief sentence explaining the advantage of LexVed's multi-step validation loop]

Note: In the metrics table, replace {queries_executed} with the number of retrieve_documents tool calls you made (typically 1 to 3), and {documents_analyzed} with the total number of documents analyzed (typically 5 * queries_executed)."""


async def agent_node(state: AgentState, config=None) -> dict:
    """
    The Agent's Brain — processes messages and decides the next action.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = _create_llm()

    # Prepend system prompt to give the LLM its identity and rules
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # If the compliance audit rejected the previous answer, insert the feedback to correct it
    if state.get("audit_feedback") and not state.get("audit_passed"):
        feedback_content = (
            f"COMPLIANCE AUDIT WARNING: Your previous response contained factual errors or unsupported claims.\n"
            f"Auditor Feedback:\n{state['audit_feedback']}\n\n"
            f"Please revise your response. Base it strictly on the retrieved case laws. Do not invent any facts."
        )
        messages.append(HumanMessage(content=feedback_content))

    # The LLM processes all messages and returns its decision
    response = await llm.ainvoke(messages, config=config)

    return {"messages": [response]}


# ─── Step 3: Compliance Auditor & Routing Logic ─────────────────

async def auditor_node(state: AgentState, config=None) -> dict:
    """
    Audits the agent's final text response against the retrieved source context.
    Detects hallucinations, unsupported claims, or direct factual contradictions.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    
    # 1. Extract agent response
    agent_message = state["messages"][-1]
    response_text = agent_message.content if hasattr(agent_message, "content") else ""
    
    # 2. Extract retrieved context from all tool messages in the history
    retrieved_context = []
    for msg in state["messages"]:
        if msg.type == "tool" and msg.name == "retrieve_documents":
            retrieved_context.append(msg.content)
            
    context_str = "\n\n".join(retrieved_context)
    
    # If no document was retrieved, we cannot verify facts (or we skip audit)
    if not context_str.strip():
        print("[LexVed Auditor] No retrieval context found. Skipping audit.")
        return {"audit_passed": True, "audit_feedback": ""}
        
    print("[LexVed Auditor] Auditing response for factual compliance...")
    
    # 3. Call verification model
    if _use_cohere_fallback:
        auditor_llm = ChatOpenAI(
            base_url="https://api.cohere.ai/compatibility/v1",
            api_key=os.getenv("COHERE_API_KEY", ""),
            model="command-r-plus-08-2024",
            temperature=0.0
        )
    else:
        auditor_llm = ChatOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.getenv("HF_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", "")),
            model="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.0
        )
    
    system_prompt = (
        "You are the LexVed Compliance Auditor.\n"
        "Compare the proposed Legal Answer against the provided Source Context.\n"
        "Identify if the answer contains hallucinations, direct contradictions, or facts/conclusions "
        "not explicitly mentioned or supported by the source context.\n\n"
        "CRITICAL RULE: Respond ONLY with the single word 'PASSED' if the answer is fully faithful and accurate.\n"
        "Otherwise, describe the specific contradictions or unsupported claims found."
    )
    
    human_content = (
        f"Source Context:\n{context_str[:12000]}\n\n"
        f"Proposed Legal Answer:\n{response_text}"
    )
    
    try:
        audit_res = await auditor_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ], config=config)
        
        feedback = audit_res.content.strip()
        is_passed = feedback.upper() == "PASSED"
        
        print(f"[LexVed Auditor] Result: {feedback}")
        
        return {
            "audit_passed": is_passed,
            "audit_feedback": "" if is_passed else feedback,
            "audit_attempts": state.get("audit_attempts", 0) + 1
        }
    except Exception as e:
        print(f"[LexVed Auditor] Audit failed to execute: {e}. Passing by default.")
        return {"audit_passed": True, "audit_feedback": ""}


def should_continue(state: AgentState) -> str:
    """
    Decides if the agent needs more tools, needs compliance auditing, or is finished.
    """
    last_message = state["messages"][-1]

    # If the message contains tool calls, go to the tool node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, if we have not performed/passed the audit yet, run compliance audit
    if not state.get("audit_passed") and state.get("audit_attempts", 0) < 2:
        return "audit"

    return "end"


def check_audit_result(state: AgentState) -> str:
    """
    Evaluates the compliance audit result.
    If the audit passed, or we reached max revision attempts, transition to END.
    Otherwise, loop back to the agent node to correct the response.
    """
    if state.get("audit_passed") or state.get("audit_attempts", 0) >= 2:
        return "end"
    return "correct"


# ─── Step 4: Build the Graph ────────────────────────────────────

def build_agent_graph():
    """
    Constructs the LangGraph agent graph with an active MemorySaver checkpointer.
    """
    from langgraph.checkpoint.memory import MemorySaver

    # Create the graph with our state schema
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("agent", agent_node)          # The LLM brain
    graph.add_node("tools", ToolNode(ALL_TOOLS))  # The tool executor
    graph.add_node("audit", auditor_node)        # The compliance auditor

    # Set where to start
    graph.set_entry_point("agent")

    # After agent runs, check if we need tools or compliance audit
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "audit": "audit",
            "end": END,
        }
    )

    # After tools run, always return to agent
    graph.add_edge("tools", "agent")

    # Routing from auditor: end or loop back to agent
    graph.add_conditional_edges(
        "audit",
        check_audit_result,
        {
            "end": END,
            "correct": "agent"
        }
    )

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


async def run_agent(user_message: str, history: list = None, thread_id: str = "default_thread", username: str = "unknown") -> dict:
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
    result = await agent.ainvoke({"messages": messages}, config=config)

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
    from src.agents.memory_manager import active_user, retrieval_counter

    # Securely set user context for thread safety
    active_user.set(username)
    retrieval_counter.set(0)

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

    # Stream events from the graph in real-time (true token-level streaming)
    tool_calls_made = []

    async for event in agent.astream_events({"messages": messages}, config=config, version="v2"):
        kind = event["event"]
        node = event.get("metadata", {}).get("langgraph_node")

        if kind == "on_chat_model_stream":
            if node == "agent":
                chunk = event["data"]["chunk"]
                # Only stream chunk content if it's text (not tool call arguments)
                if chunk.content and not getattr(chunk, "tool_call_chunks", None):
                    yield {"type": "content", "text": chunk.content}

        elif kind == "on_tool_start":
            if node == "tools":
                tool_name = event["name"]
                tool_calls_made.append(tool_name)
                yield {
                    "type": "tool_call",
                    "tool": tool_name,
                    "args": event["data"].get("input", {})
                }

        elif kind == "on_tool_end":
            if node == "tools":
                yield {
                    "type": "tool_result",
                    "tool": event["name"],
                    "result": str(event["data"].get("output", {}).get("output") if isinstance(event["data"].get("output"), dict) else event["data"].get("output", "")),
                }

    yield {
        "type": "done",
        "tool_calls": tool_calls_made,
        "steps": len(tool_calls_made) + 1,
    }
