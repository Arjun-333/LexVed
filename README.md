# LexVed Document Ingestion and Retrieval System

## Overview
LexVed is a robust vector-search-based pipeline designed for analyzing, redacting, and retrieving information from legal documents (PDFs). It processes legal documents by chunking the text intelligently based on citation patterns, redacting Personally Identifiable Information (PII) and Named Entities, embedding the text semantically, and storing the embeddings in Qdrant for fast similarity search. It also integrates with Google's Gemini 1.5 Pro Large Language Model to provide context-aware answers to user queries based on the retrieved documents.

## Architecture & Workflow
The system operates in two main phases: Ingestion and Retrieval/Generation.

### 1. Data Ingestion Pipeline
- **Extraction:** Reads PDF files from the `data/` directory using PyMuPDF.
- **Chunking:** Splits the extracted text into manageable chunks. It uses advanced regular expressions to ensure that legal citations and sentence boundaries are preserved within the chunks.
- **Redaction:** Applies Named Entity Recognition (NER) using `dslim/bert-base-NER` to detect and redact names. It also uses regex patterns to redact sensitive information like phone numbers, Aadhaar, PAN, emails, and bank accounts.
- **Embedding:** Converts the redacted text chunks into 768-dimensional dense vector embeddings using the `sentence-transformers/multi-qa-mpnet-base-cos-v1` model.
- **Storage:** Upserts the generated vectors and their corresponding metadata (source file, page number, text) into a properly structured Qdrant Vector Database.

### 2. Retrieval & Generation Pipeline
- **Query Embedding:** Converts the user's natural language query into a vector embedding using the identical local `SentenceTransformer` model.
- **Vector Search:** Queries the Qdrant database to find the top-K most semantically similar chunks based on cosine similarity.
- **Answer Generation:** Feeds the user query alongside the retrieved text chunks (context) into Google's `gemini-1.5-pro` model to generate an accurate, context-bound answer.

## File Structure & Responsibilities

### Root Directory
- `test_qdrant.py`: A simple test script to verify connection to the Qdrant database and initialize the vector collection.
- `test_embedding_qdrant.py`: The main test pipeline script. It iterates over the `data/` directory, extracts chunks, redacts PII, generates embeddings, and uploads them to Qdrant.
- `test_ingestion.py`: A quick test script for validating chunk extraction and redaction logic without running the embedding or database upload processes.
- `requirements.txt`: Contains all the Python dependencies required to run the project.

### `src/` Directory
The `src/` directory contains the core modules of the project, organized symmetrically by functionality.

#### `src/ingestion/`
- `pdf_processor.py`: Contains the logic for extracting text from PDFs (`fitz`), chunking it based on citation regex, and redacting sensitive data using NER (`transformers.pipeline`) and regex.
- `embedder.py`: Houses the `get_embeddings` function which utilizes the `SentenceTransformer` model to convert text chunks into numerical vectors.
- `uploader.py`: Contains `upload_to_qdrant`, which properly wraps vectors into Qdrant `PointStruct` objects with unique UUIDs and performs batched uploads to the vector database.

#### `src/retrieval/`
- `retriever.py`: Contains the `retrieve` function which takes a user query, embeds it, and performs a similarity search against the Qdrant collection to return the most relevant document chunks.

#### `src/generation/`
- `generator.py`: Contains the `generate_answer` function. It formulates a prompt containing the retrieved context and the user query, and sends it to the Gemini API to receive a coherent answer.

#### `src/utils/`
- `qdrant_client.py`: Configures the connection to the Qdrant server (`localhost:6333` by default) and defines the `init_collection` function to set up the collection with the correct vector dimensions and distance metric.

## Prerequisites & Setup

1. **Python Environment:** Create and activate a Python virtual environment (e.g., Python 3.9+).
2. **Install Dependencies:** Run `pip install -r requirements.txt`.
3. **Qdrant Database:** The system requires Qdrant to be running. You must start a Qdrant server instance, typically via Docker:
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```
4. **API Keys:** You need a Gemini API Key for the answer generation phase. Export it as an environment variable in your terminal:
   - Windows (Command Prompt): `set GEMINI_API_KEY=your_key_here`
   - Windows (PowerShell): `$env:GEMINI_API_KEY="your_key_here"`
   - Linux/Mac: `export GEMINI_API_KEY="your_key_here"`

## Usage Guide
1. Place the target PDF documents inside the `data/` or `data/PDF/` directory.
2. Ensure your Qdrant server is running via Docker on port 6333.
3. Run `python test_qdrant.py` to confirm database connectivity and initialize the vector collection.
4. Run `python test_embedding_qdrant.py` to process the PDFs, generate embeddings, and populate the Qdrant database.
5. Utilize the functions in `src/retrieval/retriever.py` and `src/generation/generator.py` within your application or evaluation scripts to query the system.
