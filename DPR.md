# LexVed: Detailed Project Report (DPR)

## 1. Executive Summary
LexVed is an advanced legal technology platform that uses Artificial Intelligence to analyze, sanitize, and retrieve information from legal documents. It specifically addresses the needs of legal professionals who require high-precision search and verifiable citations while maintaining strict data privacy for their clients.

## 2. The Core Problem
Legal research typically involves sifting through thousands of pages of precedents and case laws. Standard search tools often:
1. Provide irrelevant results from unrelated legal fields.
2. Leak sensitive client data (PII) to the cloud.
3. Offer no exact page references for verify facts.

## 3. The LexVed Solution: Process and Workflow

The project is structured into three primary "Pillars of Intelligence":

### Pillar 1: Intelligent Ingestion (The Reader)
When a document (PDF) is uploaded, LexVed doesn't just "read" it; it processes it:
- **Redaction**: Using Named Entity Recognition (NER), the system finds and hides names, Aadhaar numbers, and PAN cards to protect privacy.
- **Categorization**: Documents are automatically filed into the correct "legal drawer" (Criminal or Civil) based on where they are stored and what they contain.
- **Vectorization**: The text is converted into a 768-dimensional mathematical coordinate (Embeddings). This allows the AI to understand "meaning" rather than just matching words.

### Pillar 2: Hybrid Retrieval (The Searcher)
When a user asks a question, LexVed performs a sophisticated search:
- **Sub-indexing**: If you ask a criminal law question, LexVed ignores the "Civil drawer" entirely. This prevents confusion and spikes search speed.
- **Semantic + Keyword (Hybrid)**: It looks for both the "vibe" of your question and exact statute numbers (e.g., "Section 302") simultaneously.
- **Page-Aware Sorting**: It ensures that if multiple pages have relevant info, they are presented in a logical, chronological order.

### Pillar 3: Generation (The Author)
Finally, Google's Gemini LLM writes the answer:
- **Strict Grounding**: The AI is forbidden from "hallucinating" (making things up). It can *only* use the text found in your documents.
- **Mandatory Citations**: Every fact is followed by a citation like `[Source: case_ABC.pdf, Page: 5]`, making the AI's answer legally verifiable.

## 4. Technical Techniques Utilized
- **Qdrant Vector Database**: A high-performance database optimized for mathematical searches and "pre-filtering" (filtering before searching).
- **Sentence-Transformers**: The mathematical engine that turns legal jargon into vectors.
- **Named Entity Recognition (BERT)**: An AI model specifically trained to identify people, locations, and organizations in text.
- **Prompt Engineering**: Specialized instructions given to the LLM to ensure it behaves like a disciplined legal assistant.

## 5. Summary for Non-Technical Stakeholders
Imagine a library where the librarian has read every book, hidden all the private details with a black marker, and organized everything into specific rooms (Criminal Room, Civil Room). When you ask a question, the librarian doesn't just hand you a book; they write a summary for you and point exactly to the page numbers where they found the information. That is LexVed.
