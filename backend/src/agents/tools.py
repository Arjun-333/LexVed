"""
LexVed Agent Tools — Wrapping Existing Pipeline Functions as Agent Tools

HOW TOOLS WORK:
    A "tool" in LangChain is just a Python function decorated with @tool.
    The decorator does two things:
    1. Registers the function's name and docstring so the LLM can see it
    2. Handles input/output serialization automatically

    The LLM reads the docstring to decide WHEN to use each tool.
    That's why the docstrings are written like instructions — they're literally
    the LLM's guide for when to call each function.

WHAT WE'RE WRAPPING:
    We're NOT rewriting any LexVed logic. We're wrapping the existing functions
    from src/retrieval/retriever.py, src/ingestion/pdf_processor.py, etc.
    into a format that the LangGraph agent can call.

    Your existing pipeline:
        retrieve() → rerank → generate_answer()

    Now becomes tools the agent can choose from:
        retrieve_documents()  — calls your existing retrieve()
        extract_citations()   — regex + NER from your existing code
        extract_entities()    — calls SpaCy NER you already have
        deidentify_text()     — calls your existing redact functions
"""

import re
from langchain_core.tools import tool


# ─── Tool 1: Document Retrieval ──────────────────────────────────
#
# This wraps your existing src/retrieval/retriever.py → retrieve()
# which already does:
#   1. Dense embedding search (Qdrant/Pinecone)
#   2. BM25 sparse retrieval
#   3. Reciprocal Rank Fusion (RRF)
#   4. CrossEncoder reranking
#
# So this single tool call triggers your ENTIRE enhanced pipeline.

@tool
def retrieve_documents(query: str) -> str:
    """Search the legal knowledge base for relevant case law, statutes, and legal precedents.

    Use this tool when:
    - The user asks a legal question
    - The user mentions a specific law, act, or section
    - The user asks about court rulings or judgments
    - You need factual legal context to answer

    Do NOT use this tool for:
    - Simple greetings or chitchat
    - Questions about LexVed itself
    - Requests to anonymize or redact text (use deidentify_text instead)

    Args:
        query: The legal question or search query

    Returns:
        Relevant legal text passages with source metadata
    """
    from src.retrieval.retriever import retrieve

    docs, latency = retrieve(query, top_k=5)

    if not docs:
        return "No relevant documents found in the legal knowledge base."

    # Format results with source attribution (no hardcoding)
    formatted = []
    for i, doc in enumerate(docs, 1):
        payload = doc.payload
        text = payload.get("text", "")
        source = payload.get("source", "Unknown")
        page = payload.get("page", "?")
        # Extract just the filename from the full path
        source_name = source.split("/")[-1] if "/" in source else source
        formatted.append(
            f"[Source {i}: {source_name}, Page {page}]\n{text}"
        )

    result = "\n\n---\n\n".join(formatted)
    return f"Retrieved {len(docs)} documents (latency: {latency:.3f}s):\n\n{result}"


# ─── Tool 2: Legal Citation Extraction ──────────────────────────
#
# This extracts legal citations (Section numbers, Article references,
# case law citations) from a block of text.
# Uses regex patterns + SpaCy NER from your existing pipeline.

@tool
def extract_citations(text: str) -> str:
    """Extract legal citations, section references, and case law mentions from text.

    Use this tool when:
    - The user asks about specific sections or articles of law
    - You need to identify which laws are referenced in a document
    - The user wants a list of all citations in a passage

    Args:
        text: The legal text to extract citations from

    Returns:
        A structured list of all legal citations found
    """
    patterns = [
        # Indian legal citation patterns
        (r"Section\s+\d+[A-Za-z]*(?:\(\d+\))?(?:\([a-z]\))?", "Section Reference"),
        (r"Sec\.\s+\d+[A-Za-z]*", "Section Reference"),
        (r"Article\s+\d+[A-Za-z]*(?:\(\d+\))?", "Constitutional Article"),
        (r"Order\s+[IVXLCDM]+\s+Rule\s+\d+", "CPC Order/Rule"),
        (r"\d+\s+Cr\.?\s*L\.?\s*J\.?\s*\d+", "Criminal Law Journal"),
        (r"AIR\s+\d{4}\s+\w+\s+\d+", "AIR Citation"),
        (r"SCC\s+\d+", "SCC Citation"),
        (r"\(\d{4}\)\s+\d+\s+SCC\s+\d+", "SCC Full Citation"),
        (r"IPC|CrPC|CPC|IEA|NI Act", "Act Abbreviation"),
        (r"Act,?\s+\d{4}", "Legislation Year"),
    ]

    found = []
    seen = set()

    for pattern, citation_type in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            normalized = m.strip()
            if normalized.lower() not in seen:
                seen.add(normalized.lower())
                found.append(f"  - [{citation_type}] {normalized}")

    if not found:
        return "No legal citations found in the provided text."

    return f"Found {len(found)} legal citations:\n" + "\n".join(found)


# ─── Tool 3: Named Entity Recognition (NER) ─────────────────────
#
# This wraps your existing SpaCy NER pipeline from pdf_processor.py
# to extract persons, organizations, locations, dates from legal text.

@tool
def extract_entities(text: str) -> str:
    """Extract named entities (people, organizations, courts, locations, dates) from legal text.

    Use this tool when:
    - The user asks about parties involved in a case
    - The user wants to know which courts are mentioned
    - You need to identify judges, petitioners, or respondents
    - The user asks "who" or "where" questions about legal content

    Args:
        text: The legal text to analyze for entities

    Returns:
        A categorized list of all named entities found
    """
    from src.ingestion.pdf_processor import nlp

    doc = nlp(text[:10000])  # SpaCy limit for performance

    entities_by_type = {}
    for ent in doc.ents:
        label = ent.label_
        entity_text = ent.text.strip()
        if len(entity_text) < 2:
            continue
        if label not in entities_by_type:
            entities_by_type[label] = set()
        entities_by_type[label].add(entity_text)

    if not entities_by_type:
        return "No named entities found in the provided text."

    # Map SpaCy labels to human-readable names
    label_names = {
        "PERSON": "People/Judges",
        "ORG": "Organizations/Courts",
        "GPE": "Locations/Jurisdictions",
        "DATE": "Dates",
        "LAW": "Legal References",
        "NORP": "Groups/Nationalities",
        "CARDINAL": "Numbers/Quantities",
        "ORDINAL": "Ordinal References",
    }

    result_lines = []
    for label, entities in sorted(entities_by_type.items()):
        display_name = label_names.get(label, label)
        result_lines.append(f"\n{display_name}:")
        for entity in sorted(entities):
            result_lines.append(f"  - {entity}")

    return f"Found entities across {len(entities_by_type)} categories:" + "\n".join(result_lines)


# ─── Tool 4: De-identification / Redaction ───────────────────────
#
# This wraps your existing redaction functions from pdf_processor.py
# SpaCy-based name redaction + regex for phone/Aadhaar/PAN/email.

@tool
def deidentify_text(text: str) -> str:
    """Anonymize and redact personally identifiable information (PII) from legal text.

    Use this tool when:
    - The user asks to anonymize or redact a document
    - The user wants to remove personal information
    - The user needs a privacy-compliant version of text
    - The text contains names, phone numbers, Aadhaar numbers, PAN, or email addresses

    Args:
        text: The text containing personal information to redact

    Returns:
        The text with all PII replaced by redaction markers
    """
    from src.ingestion.pdf_processor import redact_names_spacy, redact_sensitive_info

    # Step 1: SpaCy-based name redaction
    redacted = redact_names_spacy(text)

    # Step 2: Regex-based PII redaction (phone, Aadhaar, PAN, email)
    redacted = redact_sensitive_info(redacted)

    # Count what was redacted
    redaction_types = {
        "[REDACTED_NAME]": redacted.count("[REDACTED_NAME]"),
        "[REDACTED_PHONE]": redacted.count("[REDACTED_PHONE]"),
        "[REDACTED_AADHAAR]": redacted.count("[REDACTED_AADHAAR]"),
        "[REDACTED_PAN]": redacted.count("[REDACTED_PAN]"),
        "[REDACTED_EMAIL]": redacted.count("[REDACTED_EMAIL]"),
    }

    active_redactions = {k: v for k, v in redaction_types.items() if v > 0}

    summary = ""
    if active_redactions:
        summary = "\n\nRedaction Summary:\n" + "\n".join(
            f"  - {k}: {v} instance(s)" for k, v in active_redactions.items()
        )
    else:
        summary = "\n\n(No PII detected in the provided text.)"

    return redacted + summary


# ─── Tools 5 & 6: Long-Term Memory (Case History & User Preferences) ───

@tool
def remember_legal_fact(fact_key: str, fact_value: str) -> str:
    """Store a key fact, detail, evidence, or user preference in the long-term memory.

    Use this tool when:
    - You discover a key fact about a client, petitioner, or respondent (e.g. client name, opposing party)
    - The user states a specific preference (e.g. "I want summarized rulings" or "use verbose language")
    - You uncover case numbers, court dates, or active legal status of an ongoing case file
    - The user tells you to remember something

    Args:
        fact_key: The classification label of the fact (e.g. 'client_name', 'opposing_counsel', 'user_preference_tone', 'case_status')
        fact_value: The detail to remember (e.g. 'Balbir Kaur', 'Steel Authority of India', 'concise', 'active')

    Returns:
        A confirmation message indicating that the fact was successfully remembered
    """
    from src.agents.memory_manager import store_user_fact
    store_user_fact(fact_key, fact_value)
    return f"Successfully saved to long-term memory: '{fact_key}' -> '{fact_value}'."


@tool
def recall_legal_facts() -> str:
    """Retrieve all stored long-term facts, case details, and preferences for the current session.

    Use this tool when:
    - A conversation starts or when you need to align with user preferences
    - You need to recall previously stored client details, opposing counsel, case states, or facts
    - The user asks "what do you know about my case?" or "do you remember my preferences?"

    Returns:
        A formatted string of all remembered facts or an empty message if nothing is saved
    """
    from src.agents.memory_manager import retrieve_user_facts
    facts = retrieve_user_facts()
    if not facts:
        return "No long-term facts or preferences are currently remembered for this user context."
    
    formatted = [f"  - {k}: {v}" for k, v in facts.items()]
    return f"Recalled {len(facts)} long-term facts and preferences:\n" + "\n".join(formatted)


# ─── Tool Registry ───────────────────────────────────────────────
#
# This list is what gets passed to the LLM. The agent sees ALL these
# tools and their descriptions, then decides which ones to call.
#
# ORDER MATTERS: The LLM tends to favor tools listed first,
# so we put the most commonly needed tool (retrieval) first.

ALL_TOOLS = [
    retrieve_documents,
    extract_citations,
    extract_entities,
    deidentify_text,
    remember_legal_fact,
    recall_legal_facts,
]
