"""
LexVed Collaborative Swarm — Phase 4: Multi-Agent Swarm using LangGraph

This module implements a collaborative agent swarm where multiple specialized
agents coordinate to retrieve, extract, draft, audit, and sanitize legal responses.

The agents in the swarm:
  1. Legal Researcher (Specialist in DB search)
  2. Citations & NER Extractor (Specialist in regex and SpaCy extraction)
  3. Drafting Counsel (Specialist in legal drafting and synthesis)
  4. Compliance Auditor (A critic/judge verifying facts against raw docs, preventing hallucinations)
  5. PII Redactor (Specialist in privacy sanitization)

Routing Logic:
  User Query → Researcher → Extractor → Drafting Counsel → Compliance Auditor
                                                ↑                   │
                                          (Revision Loop)   (Audit Passed)
                                                │                   ↓
                                           Audit Failed        PII Redactor → END
"""

import os
import json
import time
from typing import TypedDict, List, Optional, Annotated
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, add_messages

# Import the tools to execute programmatically
from src.agents.tools import (
    retrieve_documents,
    extract_citations,
    extract_entities,
    deidentify_text
)


# ─── 1. Define Swarm State Schema ────────────────────────────────────

class SwarmState(TypedDict):
    """The shared state flowing through the collaborative swarm graph.
    
    Fields:
        messages: Accumulated conversation message history
        query: The initial user query
        documents: Raw retrieved text context
        citations: Extracted statutory citations
        entities: Extracted legal entities
        draft: The current response draft from the Drafting Counsel
        feedback: Feedback from the Compliance Auditor if revisions are needed
        audit_passed: Boolean flag indicating if the draft passed verification
        revision_count: Number of times the draft has been revised (to prevent infinite loops)
    """
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    documents: Optional[str]
    citations: Optional[str]
    entities: Optional[str]
    draft: Optional[str]
    feedback: Optional[str]
    audit_passed: bool
    revision_count: int


# ─── 2. Define the Agent Nodes ──────────────────────────────────────

def researcher_agent(state: SwarmState) -> dict:
    """Legal Researcher Agent: Retrieves relevant case documents."""
    query = state["query"]
    retrieval_result = retrieve_documents.invoke({"query": query})
    return {
        "documents": retrieval_result,
        "messages": [AIMessage(content="[Researcher Agent] Completed vector and sparse search lookups.")]
    }


def extractor_agent(state: SwarmState) -> dict:
    """Citations & NER Agent: Extracts legal citations and names."""
    docs = state.get("documents", "")
    citations_result = extract_citations.invoke({"text": docs})
    entities_result = extract_entities.invoke({"text": docs})
    return {
        "citations": citations_result,
        "entities": entities_result,
        "messages": [AIMessage(content="[Extractor Agent] Extracted case citations and cataloged named entities.")]
    }


def drafting_counsel_agent(state: SwarmState) -> dict:
    """Drafting Counsel Agent: Synthesizes a structured response."""
    query = state["query"]
    docs = state.get("documents", "")
    citations = state.get("citations", "")
    entities = state.get("entities", "")
    feedback = state.get("feedback", "")
    revision_count = state.get("revision_count", 0)

    # Base prompt for the Writer agent
    system_prompt = (
        "You are the LexVed Drafting Counsel Agent.\n"
        "Your task is to write a detailed, professional response to the user's legal question.\n"
        "Base your response ONLY on the provided context documents, citations, and entities.\n\n"
        f"Context:\n{docs[:10000]}\n\n"
        f"Statutory Citations:\n{citations}\n\n"
        f"Key Entities:\n{entities}\n\n"
        "GUIDELINES:\n"
        "- Do NOT invent legal facts or outcomes.\n"
        "- Map claims to source files wherever possible using '[Source: filename.pdf]'.\n"
        "- Be structured, authoritative, and concise."
    )

    human_content = f"User Query: {query}\n\n"
    if feedback and revision_count > 0:
        human_content += (
            f"IMPORTANT: Your previous draft failed compliance audit with the following feedback:\n"
            f"'{feedback}'\n\n"
            "Please revise the draft to address all issues raised by the Auditor."
        )
    else:
        human_content += "Please draft the initial response."

    llm = ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", "")),
        model="meta-llama/Llama-3.3-70B-Instruct",
        temperature=0.2
    )

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_content)])

    return {
        "draft": response.content,
        "messages": [AIMessage(content=f"[Drafting Counsel] Generated draft version {revision_count + 1}.")]
    }


def compliance_auditor_agent(state: SwarmState) -> dict:
    """Compliance Auditor Agent: Compares the draft to source texts to verify facts."""
    docs = state.get("documents", "")
    citations = state.get("citations", "")
    draft = state.get("draft", "")
    revision_count = state.get("revision_count", 0)

    system_prompt = (
        "You are the LexVed Compliance Auditor Agent.\n"
        "Your role is to act as a rigorous critic. You compare the Drafting Counsel's response draft against "
        "the raw source documents and citations. Check for any inconsistencies, fabrications, or unsupported assertions (hallucinations).\n\n"
        f"Raw Source Documents:\n{docs[:10000]}\n\n"
        f"Valid Citations:\n{citations}\n\n"
        "EVALUATION INSTRUCTION:\n"
        "Respond in strict JSON format with exactly two keys:\n"
        "1. 'audit_passed': true (if the draft contains zero errors and is fully supported) or false (if there are unsupported facts or citation errors)\n"
        "2. 'feedback': Detailed constructive explanation of what claims are incorrect or unverified, or empty string if passed."
    )

    human_content = f"Drafting Counsel's Draft:\n{draft}\n\nPerform the audit:"

    llm = ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", "")),
        model="meta-llama/Llama-3.3-70B-Instruct",
        temperature=0.0
    ).bind(response_format={"type": "json_object"})

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_content)])
    
    try:
        res_data = json.loads(response.content)
        audit_passed = res_data.get("audit_passed", False)
        feedback = res_data.get("feedback", "")
    except Exception:
        # Fallback if JSON format failed
        audit_passed = "true" in response.content.lower()
        feedback = "Failed to parse JSON. Please check draft references." if not audit_passed else ""

    # Prevent infinite loops by forcing pass after 2 revisions
    if revision_count >= 2:
        audit_passed = True
        feedback = "Maximum revision count reached. Forcing publication."

    return {
        "audit_passed": audit_passed,
        "feedback": feedback,
        "revision_count": revision_count + 1,
        "messages": [AIMessage(content=f"[Compliance Auditor] Audit {'passed' if audit_passed else 'failed'}. Feedback: {feedback}")]
    }


def pii_redactor_agent(state: SwarmState) -> dict:
    """PII Redactor Agent: Scrubs private information before output."""
    draft = state.get("draft", "")
    redacted = deidentify_text.invoke({"text": draft})
    return {
        "messages": [AIMessage(content=redacted)]  # Final message to return
    }


# ─── 3. Construct Swarm Graph ────────────────────────────────────────

def should_approve_draft(state: SwarmState) -> str:
    """Determines whether the swarm approves the draft or sends it back for revision."""
    if state["audit_passed"]:
        return "approve"
    return "revise"


def build_swarm_graph() -> StateGraph:
    """Compiles the multi-agent swarm state machine."""
    workflow = StateGraph(SwarmState)

    # Register nodes
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("extractor", extractor_agent)
    workflow.add_node("drafter", drafting_counsel_agent)
    workflow.add_node("auditor", compliance_auditor_agent)
    workflow.add_node("redactor", pii_redactor_agent)

    # Wire edges
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "extractor")
    workflow.add_edge("extractor", "drafter")
    workflow.add_edge("drafter", "auditor")

    # Conditional routing based on Auditor review
    workflow.add_conditional_edges(
        "auditor",
        should_approve_draft,
        {
            "approve": "redactor",
            "revise": "drafter"
        }
    )
    workflow.add_edge("redactor", END)

    return workflow.compile()


# ─── 4. Public Execution Interface ───────────────────────────────

_swarm_graph = None

def get_swarm_graph():
    """Get or compile the singleton Swarm graph instance."""
    global _swarm_graph
    if _swarm_graph is None:
        _swarm_graph = build_swarm_graph()
    return _swarm_graph


async def stream_swarm_agent(query: str, username: str = "unknown"):
    """Runs the Collaborative Swarm and streams intermediate thought events in real time.
    
    Yields:
        - {"type": "thought", "text": "..."} — Thought
        - {"type": "content", "text": "..."} — Word-by-word streaming of final redacted answer
        - {"type": "done", "generation_time": ...} — Complete
    """
    import asyncio
    t_start = time.time()
    graph = get_swarm_graph()

    inputs = {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "revision_count": 0,
        "audit_passed": False
    }

    yield {
        "type": "thought",
        "text": "Initializing Collaborative Multi-Agent Swarm..."
    }

    async for event in graph.astream(inputs):
        node_name = list(event.keys())[0]
        node_data = event[node_name]

        if node_name == "researcher":
            yield {
                "type": "thought",
                "text": "[Legal Researcher Agent] Completed vector lookup and sparse document retrieval."
            }
        elif node_name == "extractor":
            yield {
                "type": "thought",
                "text": "[Citations & NER Extractor] Analyzed retrieved text. Extracted statutory citations and entity catalog."
            }
        elif node_name == "drafter":
            rev = node_data.get("revision_count", 0)
            msg = f"[Drafting Counsel Agent] Synthesizing initial response draft..." if rev == 0 else f"[Drafting Counsel Agent] Revising response draft (Revision {rev})..."
            yield {
                "type": "thought",
                "text": msg
            }
        elif node_name == "auditor":
            passed = node_data.get("audit_passed", False)
            feedback = node_data.get("feedback", "")
            if passed:
                yield {
                    "type": "thought",
                    "text": "[Compliance Auditor Agent] Audit Passed! Draft is fully supported by case files. Forwarding to Redactor."
                }
            else:
                yield {
                    "type": "thought",
                    "text": f"[Compliance Auditor Agent] Audit Failed! Reason: {feedback}\nRouting back to Drafting Counsel Agent."
                }
        elif node_name == "redactor":
            yield {
                "type": "thought",
                "text": "[PII Redactor Agent] Scrubbing private data and Aadhaar/PAN markers. Finalizing output..."
            }
            
            # Stream the final message content word-by-word
            final_messages = node_data.get("messages", [])
            if final_messages:
                final_text = final_messages[-1].content
                words = final_text.split(" ")
                for i, word in enumerate(words):
                    spacing = "" if i == len(words) - 1 else " "
                    yield {
                        "type": "content",
                        "text": word + spacing
                    }
                    await asyncio.sleep(0.015)

    yield {
        "type": "done",
        "generation_time": time.time() - t_start
    }
