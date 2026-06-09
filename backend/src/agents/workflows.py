"""
LexVed Workflows — Phase 3: Multi-Step Workflow Automation using LangGraph

This module implements structured, deterministic workflows (automated DAGs)
for complex legal operations that require multiple sequential tasks.

Instead of relying on the LLM to choose tools dynamically (which can be flaky
for multi-stage procedures), we map out the workflow nodes explicitly in a
LangGraph, using the LLM only for legal drafting/synthesis.

The primary workflow is:
    User Request (Case/Topic) 
           ↓
    1. research_node       (Auto-Retrieval from Qdrant/Pinecone)
           ↓
    2. extraction_node     (Auto-Citations & NER Extraction)
           ↓
    3. draft_node          (Structured Legal Brief Synthesis)
           ↓
    4. sanitization_node   (PII Redaction / Compliance Scrubbing)
           ↓
    Final Anonymized Case Brief
"""

import os
from typing import TypedDict, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Import the tools to execute programmatically
from src.agents.tools import (
    retrieve_documents,
    extract_citations,
    extract_entities,
    deidentify_text
)


# ─── 1. Define the State Schema ───────────────────────────────────────

class BriefWorkflowState(TypedDict):
    """The state schema flowing through the Case Brief generation workflow.
    
    Fields:
        query: The initial legal query or case name requested by the user
        username: The active username to track memory context
        raw_documents: The merged text context retrieved from the databases
        citations: List of legal statutes/precedents extracted from context
        entities: List of parties, judges, and courts extracted from context
        draft_brief: The initial structured case brief drafted by the LLM
        final_brief: The sanitized, PII-redacted final case brief
    """
    query: str
    username: Optional[str]
    raw_documents: Optional[str]
    citations: Optional[str]
    entities: Optional[str]
    draft_brief: Optional[str]
    final_brief: Optional[str]


# ─── 2. Define the Workflow Nodes ────────────────────────────────────

def research_node(state: BriefWorkflowState) -> dict:
    """Step 1: Retrieve relevant documents automatically based on user query."""
    query = state["query"]
    
    # Programmatically invoke retrieve_documents tool
    retrieval_result = retrieve_documents.invoke({"query": query})
    
    return {
        "raw_documents": retrieval_result
    }


def extraction_node(state: BriefWorkflowState) -> dict:
    """Step 2: Parse retrieved documents for legal citations and named entities."""
    docs = state.get("raw_documents", "")
    
    # Programmatically invoke citation and entity tools
    citations_result = extract_citations.invoke({"text": docs})
    entities_result = extract_entities.invoke({"text": docs})
    
    return {
        "citations": citations_result,
        "entities": entities_result
    }


def draft_node(state: BriefWorkflowState) -> dict:
    """Step 3: Synthesize a structured case brief using Llama 3.3 70B."""
    query = state["query"]
    docs = state.get("raw_documents", "")
    citations = state.get("citations", "")
    entities = state.get("entities", "")
    username = state.get("username", "Counsel")

    # Construct the instruction set for case brief drafting
    system_prompt = (
        f"You are the LexVed Senior Legal Briefing Counsel. Address the user directly as {username}.\n\n"
        "Your task is to draft a comprehensive and structured Case Brief based ONLY on the provided context, "
        "extracted citations, and entities. Do NOT formulate facts from memory.\n\n"
        "STRUCTURE YOUR CASE BRIEF WITH THESE SECTIONS:\n"
        "1. CASE TITLE & PARTIES (Name the petitioner, respondent, and court based on the extracted entities)\n"
        "2. STATEMENT OF FACTS (A detailed summary of the background and dispute)\n"
        "3. LEGAL QUESTIONS/ISSUES (List the core questions of law the court had to decide)\n"
        "4. RELEVANT STATUTARY PROVISIONS (List the acts, sections, or articles based on the extracted citations)\n"
        "5. HELD & RATIONALE (Explain what the court decided, the judgment, and the reasoning behind it)\n"
        "6. SIGNIFICATION (Explain the precedent value of this ruling)\n\n"
        "MANDATORY GUIDELINES:\n"
        "- Do NOT write like a generic letter (no 'Sincerely', no placeholders).\n"
        "- Every claim must link to its source file: [Source: filename.pdf, Page: X].\n"
        "- If the retrieved context is insufficient to cover a section, say so honestly."
    )

    human_content = (
        f"User Query/Case Name: {query}\n\n"
        f"Retrieved Context:\n{docs[:10000]}\n\n"
        f"Extracted Statutory Citations:\n{citations}\n\n"
        f"Extracted Legal Entities:\n{entities}\n\n"
        "Please draft the Case Brief:"
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,  # Low temperature for factual synthesis
        api_key=os.getenv("GROQ_API_KEY"),
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content)
    ]

    response = llm.invoke(messages)

    return {
        "draft_brief": response.content
    }


def sanitization_node(state: BriefWorkflowState) -> dict:
    """Step 4: Redact personally identifiable information (PII) to ensure compliance."""
    draft = state.get("draft_brief", "")
    
    # Programmatically invoke deidentify_text tool to redact names, Aadhaar, PAN, phone numbers
    redacted_result = deidentify_text.invoke({"text": draft})
    
    return {
        "final_brief": redacted_result
    }


# ─── 3. Construct the Workflow Graph ─────────────────────────────────

def build_brief_workflow() -> StateGraph:
    """Constructs the LangGraph state machine for the Case Brief workflow."""
    # Initialize the graph with the custom State Dict
    workflow = StateGraph(BriefWorkflowState)

    # Register the deterministic nodes
    workflow.add_node("research", research_node)
    workflow.add_node("extraction", extraction_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("sanitization", sanitization_node)

    # Wire the pipeline sequentially (Linear DAG)
    workflow.set_entry_point("research")
    workflow.add_edge("research", "extraction")
    workflow.add_edge("extraction", "draft")
    workflow.add_edge("draft", "sanitization")
    workflow.add_edge("sanitization", END)

    # Compile the graph
    return workflow.compile()


# ─── 4. Public Execution Interface ───────────────────────────────

_brief_workflow = None

def get_brief_workflow():
    """Get or compile the singleton Case Brief workflow instance."""
    global _brief_workflow
    if _brief_workflow is None:
        _brief_workflow = build_brief_workflow()
    return _brief_workflow


async def stream_brief_workflow(query: str, username: str = "unknown"):
    """Runs the Case Brief workflow and streams state changes/thoughts in real-time.
    
    Yields events matching the expected format of the frontend SSE client:
        - {"type": "agent_thought", "text": "..."} — Thought/Progress
        - {"type": "content", "text": "..."} — Word-by-word streaming of final brief
        - {"type": "done", "generation_time": ...} — Finish
    """
    import time
    import asyncio
    
    t_start = time.time()
    workflow = get_brief_workflow()

    # Initial state inputs
    inputs = {
        "query": query,
        "username": username
    }

    yield {
        "type": "thought",
        "text": "Initializing Automated Case Brief Workflow DAG..."
    }

    # Stream graph updates node-by-node
    async for event in workflow.astream(inputs):
        node_name = list(event.keys())[0]
        node_data = event[node_name]

        if node_name == "research":
            yield {
                "type": "thought",
                "text": "Step 1/4: Completed document retrieval from hybrid database indices."
            }
        elif node_name == "extraction":
            citations = node_data.get("citations", "")
            entities = node_data.get("entities", "")
            yield {
                "type": "thought",
                "text": "Step 2/4: Completed citation and entity extraction.\n"
                        f"- Citations: {citations[:150]}...\n"
                        f"- Entities: {entities[:150]}..."
            }
        elif node_name == "draft":
            yield {
                "type": "thought",
                "text": "Step 3/4: Completed case brief synthesis with Llama 3.3. Preparing sanitization..."
            }
        elif node_name == "sanitization":
            yield {
                "type": "thought",
                "text": "Step 4/4: Completed PII redaction and compliance checks. Preparing streaming output..."
            }
            
            # Stream the final redacted case brief word-by-word to the user interface
            final_brief = node_data.get("final_brief", "")
            words = final_brief.split(" ")
            for i, word in enumerate(words):
                spacing = "" if i == len(words) - 1 else " "
                yield {
                    "type": "content",
                    "text": word + spacing
                }
                await asyncio.sleep(0.015)

    total_time = time.time() - t_start
    yield {
        "type": "done",
        "generation_time": total_time
    }
