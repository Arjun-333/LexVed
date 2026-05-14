# ===========================================================
# ⚖️  LEGAL RAG PIPELINE — ALL EMBEDDING MODELS
#     MPNet | MiniLM | DistilBERT | E5-Mistral
#     Cohere Embed v3 | BGE-M3
# ===========================================================

import os, re, time, csv, psutil, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from transformers import logging as hf_logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

warnings.filterwarnings("ignore", message="Some weights of the model")
hf_logging.set_verbosity_error()

# ----------------------------------------------------------
# PDF library detection
# ----------------------------------------------------------
try:
    import fitz
    PDF_LIBRARY = "pymupdf"
    print("✅ Using PyMuPDF for PDF processing")
except ImportError:
    try:
        from pypdf import PdfReader
        PDF_LIBRARY = "pypdf"
        print("⚠ Using pypdf as fallback")
    except ImportError:
        print("❌ ERROR: No PDF library installed! Run: pip install PyMuPDF")
        # In this environment, we'll try to proceed or exit gracefully
        PDF_LIBRARY = None

# ----------------------------------------------------------
# NLTK resources
# ----------------------------------------------------------
import nltk
for resource in ['wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, pipeline

# ----------------------------------------------------------
# Pinecone import — support v2 and v3
# ----------------------------------------------------------
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_VERSION = "v3"
    print("✅ Using Pinecone v3+")
except ImportError:
    try:
        import pinecone
        PINECONE_VERSION = "v2"
        print("✅ Using Pinecone v2 (legacy)")
    except ImportError:
        print("❌ Pinecone not installed!")
        raise

# ----------------------------------------------------------
# LLM API (Groq)
# ----------------------------------------------------------
import requests
import tiktoken

print("✅ All imports successful!")


# ===========================================================
# 1. API KEYS & CLIENTS
# ===========================================================
# Use .env or fallback to user provided keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "[REMOVED_PINECONE_KEY]")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY",     "[REMOVED_GROQ_KEY]")
COHERE_API_KEY   = os.getenv("COHERE_API_KEY",   "[REMOVED_COHERE_KEY]")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

if PINECONE_VERSION == "v3":
    pc = Pinecone(api_key=PINECONE_API_KEY)
else:
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1")

print("✅ API clients configured")


# ===========================================================
# 2. PINECONE INDEX — created per model with correct dimension
# ===========================================================
# Dimension map per model
MODEL_DIMENSIONS = {
    "multi-qa-mpnet-base-cos-v1":    768,
    "multi-qa-MiniLM-L6-cos-v1":     384,
    "multi-qa-distilbert-cos-v1":    768,
    "E5-Mistral":                   4096,
    "Cohere-embed-english-v3":      1024,
    "BGE-M3":                       1024,
}

def get_or_create_index(index_name: str, dimension: int):
    """Create Pinecone index if it doesn't exist, then return handle."""
    if PINECONE_VERSION == "v3":
        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            print(f"Creating index '{index_name}' (dim={dimension})...")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"✅ Index '{index_name}' created.")
        else:
            print(f"✅ Index '{index_name}' already exists.")
        return pc.Index(index_name)
    else:
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, dimension=dimension, metric="cosine")
            print(f"✅ Index '{index_name}' created (dim={dimension}).")
        else:
            print(f"✅ Index '{index_name}' already exists.")
        return pinecone.Index(index_name)


# ===========================================================
# 3. NER REDACTION (optional)
# ===========================================================
USE_NER_REDACTION = False
if USE_NER_REDACTION:
    _ner_name      = "dslim/bert-base-NER"
    _ner_tokenizer = AutoTokenizer.from_pretrained(_ner_name)
    _ner_model     = AutoModelForTokenClassification.from_pretrained(_ner_name)
    ner_pipeline   = pipeline("ner", model=_ner_model, tokenizer=_ner_tokenizer,
                               aggregation_strategy="simple")
    print("✅ NER model loaded")

def run_ner_redact(text):
    if not USE_NER_REDACTION:
        return text
    try:
        ents = ner_pipeline(text)
        for e in ents:
            if e['entity_group'] == 'PER':
                text = re.sub(r'\b{}\b'.format(re.escape(e['word'].replace("##", ""))),
                              '[REDACTED]', text)
        return text
    except:
        return text


# ===========================================================
# 4. DRIVE + PDF SELECTION (ADAPTED FOR LOCAL)
# ===========================================================

# In local environment, we skip mounting drive
print("ℹ Running in Local Environment. Skipping Google Drive mount.")

folder_type = input("Folder type (Civil/Criminal/Both): ").strip().lower()
if folder_type not in ["civil", "criminal", "both"]:
    print("⚠ Invalid. Defaulting to Civil.")
    folder_type = "civil"

# Local PDF path
if folder_type == "both":
    pdf_base_dirs = [
        str(Path(__file__).parent / "data" / "PDF" / "CIVIL"),
        str(Path(__file__).parent / "data" / "PDF" / "CRIMINAL")
    ]
    folder_label = "Civil & Criminal"
else:
    pdf_base_dirs = [str(Path(__file__).parent / "data" / "PDF" / folder_type.upper())]
    folder_label = folder_type.title()

process_all = input(f"Process all {folder_label} PDFs? (y/n): ").strip().lower()

pdf_files = []
for pdf_base_dir in pdf_base_dirs:
    if process_all == "y":
        pdf_files.extend(list(Path(pdf_base_dir).rglob("*.pdf")))
    else:
        selections = input(f"Enter folder names/years for {Path(pdf_base_dir).name} (e.g., 2001,2005-2008): ").strip()
        for part in selections.split(","):
            if "-" in part:
                try:
                    start, end = map(int, part.split("-"))
                    for yr in range(start, end + 1):
                        p = Path(pdf_base_dir) / str(yr)
                        if p.exists():
                            pdf_files.extend(list(p.rglob("*.pdf")))
                        else:
                            print(f"⚠ Folder not found: {p}")
                except ValueError:
                    print(f"⚠ Invalid range: {part}")
            else:
                p = Path(pdf_base_dir) / part.strip()
                if p.exists():
                    pdf_files.extend(list(p.rglob("*.pdf")))
                else:
                    print(f"⚠ Folder not found: {p}")

if not pdf_files:
    print(f"⚠ No PDFs found in {pdf_base_dir}. Exiting.")
    sys.exit()
else:
    print(f"✅ Found {len(pdf_files)} PDFs to process.")


# ===========================================================
# 5. PDF CHUNKING
# ===========================================================
def extract_chunks(pdf_path, chunk_size=200):
    chunks = []
    citation_pattern = re.compile(
        r"(Section\s\d+[A-Za-z]|Sec\.\s\d+[A-Za-z]|\d+\s?Cr\.?\s?\d+)", re.IGNORECASE
    )
    try:
        if PDF_LIBRARY == "pymupdf":
            import fitz
            doc  = fitz.open(pdf_path)
            text = "\n".join([page.get_text("text") for page in doc])
            doc.close()
        elif PDF_LIBRARY == "pypdf":
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            text   = "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            return []

        text      = re.sub(r"[*_]", "", text)
        text      = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r'(?<=[.?!]) +|\n+', text)

        buf = ""
        for sent in sentences:
            parts = citation_pattern.split(sent)
            for part in parts:
                seg = part.strip()
                if not seg:
                    continue
                if len(buf.split()) + len(seg.split()) < chunk_size:
                    buf += " " + seg
                else:
                    if buf.strip():
                        chunks.append({'text': buf.strip(), 'source': str(pdf_path)})
                    buf = seg
        if buf.strip():
            chunks.append({'text': buf.strip(), 'source': str(pdf_path)})

        return chunks
    except Exception as e:
        print(f"⚠ Error processing {pdf_path}: {e}")
        return []

def process_pdfs(pdf_files):
    all_chunks = []
    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        chunks = extract_chunks(pdf)
        for c in chunks:
            c['text'] = run_ner_redact(c['text'])
        all_chunks.extend(chunks)
    print(f"✅ Total chunks: {len(all_chunks)}")
    return all_chunks


# ===========================================================
# 6. EMBEDDING MODEL HELPERS
# ===========================================================

# ── 6a. E5-Mistral (local HuggingFace checkpoint) ──────────
def make_e5_embedder(model_path="./hf_models/e5-mistral-7b-instruct"):
    class E5Embedder:
        def __init__(self, path):
            print(f"Loading E5-Mistral from {path} ...")
            # If local path doesn't exist, we'll try to load from Hub
            local_only = os.path.exists(path)
            if not local_only:
                print(f"⚠ Local path {path} not found. Attempting to load from HuggingFace Hub (intfloat/e5-mistral-7b-instruct)...")
                path = "intfloat/e5-mistral-7b-instruct"
            
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_only)
            # Use CPU if CUDA is not available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            self.model     = AutoModel.from_pretrained(path, local_files_only=local_only).to(device)
            self.model.eval()
            self.device = device

        def encode(self, texts, batch_size=4, show_progress_bar=True, convert_to_numpy=True):
            out      = []
            iterator = range(0, len(texts), batch_size)
            if show_progress_bar:
                iterator = tqdm(iterator, desc="E5-Mistral encoding")
            with torch.no_grad():
                for i in iterator:
                    batch   = texts[i:i + batch_size]
                    inputs  = self.tokenizer(batch, truncation=True, padding=True,
                                             return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    # Mean pooling
                    emb     = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    out.append(emb)
            return np.vstack(out)

    return E5Embedder(model_path)


# ── 6b. BGE-M3 (local via FlagEmbedding — NO fallback) ─────
def make_bge_m3_embedder():
    from FlagEmbedding import BGEM3FlagModel   # pip install FlagEmbedding

    class BGEM3Embedder:
        def __init__(self):
            print("Loading BGE-M3 via FlagEmbedding ...")
            # Use CPU if CUDA not available
            use_fp16 = torch.cuda.is_available()
            self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16)

        def encode(self, texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True):
            results = self.model.encode(
                texts,
                batch_size=batch_size,
                max_length=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            return results["dense_vecs"]   # numpy array shape (N, 1024)

    return BGEM3Embedder()


# ── 6c. Cohere Embed v3 (API) ───────────────────────────────
def make_cohere_embedder(api_key: str):
    import cohere   # pip install cohere

    class CohereEmbedder:
        def __init__(self, key: str):
            self.client = cohere.Client(api_key=key)
            self.model  = "embed-english-v3.0"
            # Validate key immediately
            print("Validating Cohere API key ...")
            try:
                self.client.embed(texts=["test"], model=self.model,
                                  input_type="search_document")
                print("✅ Cohere API key validated.")
            except Exception as e:
                print(f"❌ Cohere API key validation failed: {e}")
                raise

        # ── document encoding (corpus upsert) ──
        def encode(self, texts, batch_size=96, show_progress_bar=True,
                   convert_to_numpy=True, input_type="search_document"):
            all_emb  = []
            iterator = range(0, len(texts), batch_size)
            if show_progress_bar:
                iterator = tqdm(iterator, desc="Cohere encoding")
            for i in iterator:
                batch = texts[i:i + batch_size]
                resp  = self.client.embed(texts=batch, model=self.model,
                                          input_type=input_type)
                all_emb.extend(resp.embeddings)
            return np.array(all_emb)

        # ── query encoding ──
        def encode_query(self, texts):
            resp = self.client.embed(texts=texts, model=self.model,
                                     input_type="search_query")
            return np.array(resp.embeddings)

    return CohereEmbedder(api_key)


# ===========================================================
# 7. EMBEDDING MODEL MENU
# ===========================================================
def select_embedding_model():
    """
    Presents a menu, loads the chosen embedder, returns
    (model_name_str, embedder_obj, pinecone_index_name).
    Cohere asks for the API key before proceeding.
    """
    print("\n" + "="*55)
    print("   Select Embedding Model")
    print("="*55)
    print("  1) MPNet          (multi-qa-mpnet-base-cos-v1)")
    print("  2) MiniLM         (multi-qa-MiniLM-L6-cos-v1)")
    print("  3) DistilBERT     (multi-qa-distilbert-cos-v1)")
    print("  4) E5-Mistral     (local checkpoint)")
    print("  5) Cohere Embed v3 (API)")
    print("  6) BGE-M3         (FlagEmbedding / local)")
    print("  q) Quit")
    print("="*55)

    while True:
        choice = input("Enter choice: ").strip().lower()

        if choice == "q":
            return None, None, None

        elif choice == "1":
            name     = "multi-qa-mpnet-base-cos-v1"
            embedder = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-cos-v1")
            idx_name = "mpnet-index"

        elif choice == "2":
            name     = "multi-qa-MiniLM-L6-cos-v1"
            embedder = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
            idx_name = "minilm-index"

        elif choice == "3":
            name     = "multi-qa-distilbert-cos-v1"
            embedder = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1")
            idx_name = "distilbert-index"

        elif choice == "4":
            name     = "E5-Mistral"
            embedder = make_e5_embedder()
            idx_name = "e5-mistral-index"

        elif choice == "5":
            api_key = input(f"Enter your Cohere API key (default: {COHERE_API_KEY[:5]}...): ").strip()
            if not api_key:
                api_key = COHERE_API_KEY
            if not api_key:
                print("⚠ No key entered. Please try again.")
                continue
            try:
                embedder = make_cohere_embedder(api_key)
            except Exception as e:
                print(f"❌ Cohere init failed: {e}")
                continue
            name     = "Cohere-embed-english-v3"
            idx_name = "cohere-embed-v3-index"

        elif choice == "6":
            try:
                embedder = make_bge_m3_embedder()
            except Exception as e:
                print(f"❌ BGE-M3 init failed: {e}")
                continue
            name     = "BGE-M3"
            idx_name = "bge-m3-index"

        else:
            print("⚠ Invalid choice.")
            continue

        print(f"✅ Loaded: {name}")
        return name, embedder, idx_name


# ===========================================================
# 8. EMBED & STORE INTO PINECONE
# ===========================================================
def embed_and_store(chunks, embedder, model_name: str, index, batch_size=64):
    """Encode all chunks and upsert into the provided Pinecone index."""
    if not chunks:
        print("⚠ No chunks to embed.")
        return 0, 0

    texts = [c['text'] for c in chunks]

    # Cohere uses doc input_type; all others use .encode() directly
    is_cohere = hasattr(embedder, "encode_query")

    t0 = time.time()
    if is_cohere:
        vectors = embedder.encode(texts, show_progress_bar=True,
                                  input_type="search_document")
    else:
        vectors = embedder.encode(texts, show_progress_bar=True,
                                  batch_size=batch_size)
    embedding_time = time.time() - t0
    print(f"✅ Embedding complete in {embedding_time:.2f}s")

    # Upsert in batches
    batch = []
    for i, vec in enumerate(tqdm(vectors, desc="Upserting to Pinecone")):
        batch.append({
            'id':     f"{model_name}_{i}", # Unique ID per model
            'values': vec.tolist(),
            'metadata': {
                'text':   chunks[i]['text'],
                'source': chunks[i]['source']
            }
        })
        if len(batch) >= batch_size:
            index.upsert(vectors=batch)
            batch = []
    if batch:
        index.upsert(vectors=batch)

    print(f"✅ {len(vectors)} vectors upserted to Pinecone.")
    return embedding_time, len(vectors)


# ===========================================================
# 9. RETRIEVAL & GROQ GENERATION
# ===========================================================
def generate_with_groq(prompt: str, model="llama-3.1-8b-instant") -> str:
    if not GROQ_API_KEY:
        return "[Error: No GROQ_API_KEY]"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    for attempt in range(3):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2**attempt)
            else:
                return f"[Groq Error: {e}]"

def retrieve_answer(query, embedder, index, top_k=5):
    print(f"\n🔹 Query: {query}")

    # Encode query (Cohere needs separate method)
    is_cohere = hasattr(embedder, "encode_query")
    if is_cohere:
        q_vec = embedder.encode_query([query])[0]
    else:
        q_vec = embedder.encode([query], show_progress_bar=False)[0]

    # Pinecone query
    res   = index.query(vector=q_vec.tolist(), top_k=top_k, include_metadata=True)
    texts = [m['metadata'].get('text', '') for m in res['matches']]
    context = "\n\n".join(texts)

    prompt = f"""Answer the following query using ONLY the context below.

Context:
{context}

Query: {query}

Answer:
"""

    answer = generate_with_groq(prompt)

    if not answer:
        answer = "⚠ Model returned no answer."

    return answer, texts, q_vec


# ===========================================================
# 10. METRICS
# ===========================================================
enc = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def compute_metrics(preds, gts, retrieved_texts_list, q_embeds, embedder,
                    retrieval_times=None, generation_times=None,
                    log_file="metrics_log.csv",
                    embedding_time=0.0, index_size=0):

    df_list           = []
    rouge_scorer_obj  = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                                  use_stemmer=True)
    smoothie          = SmoothingFunction().method4
    is_cohere         = hasattr(embedder, "encode_query")

    for i, (pred, gt, texts, q_vec) in enumerate(
            zip(preds, gts, retrieved_texts_list, q_embeds)):

        r = rouge_scorer_obj.score(gt, pred)
        b = sentence_bleu([gt.split()], pred.split(), smoothing_function=smoothie)
        m = meteor_score([gt.split()], pred.split())
        
        try:
            _, _, F1 = bert_score([pred], [gt], lang='en', rescale_with_baseline=True)
            bert_f1_val = float(F1[0])
        except:
            bert_f1_val = 0.0

        # Context similarity
        if is_cohere:
            retr_vecs = embedder.encode_query(texts)
        else:
            retr_vecs = embedder.encode(texts, show_progress_bar=False) if texts else []

        sims = cosine_similarity([q_vec], retr_vecs)[0] if len(retr_vecs) > 0 else []

        # FCD
        if texts:
            try:
                _, _, F1_ctx = bert_score([pred], [" ".join(texts)], lang="en")
                fcd_value = float(1 - F1_ctx.mean())
            except:
                fcd_value = 1.0
        else:
            fcd_value = 1.0

        # Token-level stats
        gt_tokens   = set(gt.lower().split())
        pred_tokens = set(pred.lower().split())
        common      = gt_tokens & pred_tokens

        bias_value            = abs(len(gt.split()) - len(pred.split())) / max(1, len(gt.split()))
        faithfulness          = len(common) / len(gt_tokens)   if gt_tokens   else 0
        terminology_precision = len(common) / max(1, len(pred_tokens))
        coverage              = len(common) / len(gt_tokens) * 100 if gt_tokens else 0

        # Latency
        retrieval_latency  = retrieval_times[i]  if retrieval_times  and i < len(retrieval_times)  else 0
        generation_latency = generation_times[i] if generation_times and i < len(generation_times) else 0
        end_to_end_latency = retrieval_latency + generation_latency
        throughput         = round(1.0 / (end_to_end_latency + 1e-6), 2)

        cosine_sim = float(np.mean(sims)) if len(sims) > 0 else 0.0
        topk_acc   = sum(1 for s in sims if s > 0.8) / len(sims) if len(sims) > 0 else 0

        cpu_use = psutil.cpu_percent()
        ram_use = round(psutil.virtual_memory().used / (1024 ** 3), 2)

        context        = " ".join(texts)
        context_tokens = count_tokens(context)
        context_len    = sum(len(t.split()) for t in texts)

        row = {
            "GT":                       gt,
            "Answer":                   pred,
            "ROUGE-1":                  r['rouge1'].fmeasure,
            "ROUGE-2":                  r['rouge2'].fmeasure,
            "ROUGE-L":                  r['rougeL'].fmeasure,
            "BLEU":                     b,
            "METEOR":                   m,
            "BERT-F1":                  bert_f1_val,
            "FCD":                      fcd_value,
            "Bias":                     bias_value,
            "Faithfulness":             faithfulness,
            "Terminology_Precision":    terminology_precision,
            "GroundTruth_Coverage(%)":  coverage,
            "Cosine_Sim":               cosine_sim,
            "TopK_Accuracy":            topk_acc,
            "Retrieval_Latency(sec)":   retrieval_latency,
            "Generation_Latency(sec)":  generation_latency,
            "EndToEnd_Latency(sec)":    end_to_end_latency,
            "Throughput(q/s)":          throughput,
            "Context_Tokens":           context_tokens,
            "Context_Length(words)":    context_len,
            "CPU_Usage(%)":             cpu_use,
            "RAM_Usage(GB)":            ram_use,
            "Embedding_Time(sec)":      embedding_time,
            "Index_Size(vectors)":      index_size,
            "Timestamp":                time.strftime("%Y-%m-%d %H:%M:%S",
                                                       time.localtime())
        }
        df_list.append(row)

        # Live CSV logging
        write_header = not os.path.exists(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(list(row.keys()))
            writer.writerow(list(row.values()))

    df = pd.DataFrame(df_list)
    print("\n✅ Evaluation complete. Metrics saved to:", log_file)
    return df


# ===========================================================
# 11. FULL PIPELINE
# ===========================================================
def run_pipeline(pdf_files, queries, gts, embedder, model_name, index,
                 log_file="metrics_log.csv"):

    # Step 1 — chunk PDFs
    chunks = process_pdfs(pdf_files)

    # Step 2 — embed & store; get timing
    embedding_time, index_size = embed_and_store(chunks, embedder, model_name, index)

    # Step 3 — per-query retrieval + generation
    preds, retrieved_texts_list, q_embeds = [], [], []
    retrieval_times, generation_times     = [], []

    for q in queries:
        t_ret_start = time.time()
        ans, texts, q_vec = retrieve_answer(q, embedder, index)
        t_ret_end   = time.time()

        retrieval_latency  = t_ret_end - t_ret_start
        generation_latency = 0.0        # generation is inside retrieve_answer; latency captured there

        retrieval_times.append(retrieval_latency)
        generation_times.append(generation_latency)

        preds.append(ans)
        retrieved_texts_list.append(texts)
        q_embeds.append(q_vec)

        print(f"⏱️ E2E Latency: {retrieval_latency:.2f}s")

    # Step 4 — compute & log metrics
    df_metrics = compute_metrics(
        preds, gts, retrieved_texts_list, q_embeds, embedder,
        retrieval_times=retrieval_times,
        generation_times=generation_times,
        log_file=log_file,
        embedding_time=embedding_time,
        index_size=index_size
    )
    return df_metrics


# ===========================================================
# 12. QUERIES & GROUND TRUTHS
# ===========================================================
civil_queries = [
    "Can a person convicted of a criminal offence whose conviction has not been suspended be appointed as Chief Minister?",
    "What was the main legal issue regarding the promotion of Sub-Inspectors to Inspectors in Uttar Pradesh?",
    "What dispute arose regarding the Floor Area Ratio (FAR) in the auctioned property case?",
    "On what ground was the election of the returned candidate challenged?",
    "What was the central issue regarding the Income Tax Settlement Commission's order?"
]
civil_gts = [
    "No, a person convicted of a criminal offence whose conviction has not been suspended is disqualified and cannot be appointed or continue as Chief Minister.",
    "The main issue was whether promotions should be based on merit under the 1965 government order or on seniority subject to rejection of the unfit under the 1994 rules.",
    "The dispute arose because the auction and sale deed allowed a FAR of 2.0, but later building regulations reduced it to 1.75, leading to conflict over applicable development rights.",
    "The election was challenged on the ground that the returned candidate was disqualified for holding an office of profit under the Government.",
    "The issue was whether the settlement order could be declared void under Section 245D(6) on the ground that it was obtained by fraud or misrepresentation of facts."
]

criminal_queries = [
    "What is the legal principle governing the plea of alibi in criminal cases?",
    "Can prosecution continue after the omission of Section 276DD of the Income Tax Act?",
    "Can a conviction be based solely on a dying declaration?",
    "Does a Labour Court have jurisdiction to set aside an ex parte award after 30 days of its publication?",
    "What was the key evidence supporting the conviction under Section 307 IPC?"
]
criminal_gts = [
    "The plea of alibi requires proving that the accused was at a place so far away at the relevant time that it was physically impossible for them to be present at the scene of the crime.",
    "No, prosecution cannot continue after omission of Section 276DD because omission is treated as obliteration of the provision and Section 6 of the General Clauses Act does not apply to omissions.",
    "Yes, a conviction can be based solely on a dying declaration if it is found to be true, voluntary, and inspires confidence, even without corroboration.",
    "No, once 30 days have passed after publication of the award and it becomes enforceable, the Labour Court becomes functus officio and has no jurisdiction to set aside the ex parte award.",
    "The conviction was supported by the testimony of injured eyewitnesses and recovery of the knife with human blood matching the victim's blood group."
]

# Pick query set based on folder_type selected at runtime
if folder_type == "civil":
    active_queries = civil_queries
    active_gts     = civil_gts
elif folder_type == "criminal":
    active_queries = criminal_queries
    active_gts     = criminal_gts
else:
    # Combine both for "both"
    active_queries = civil_queries + criminal_queries
    active_gts     = civil_gts + criminal_gts


# ===========================================================
# 13. MAIN — model selection loop
# ===========================================================
while True:
    model_name, embedder, idx_name = select_embedding_model()

    if model_name is None:
        print("👋 Exiting.")
        break

    # Create a fresh Pinecone index for this model (correct dimension)
    dimension = MODEL_DIMENSIONS[model_name]
    index     = get_or_create_index(idx_name, dimension)

    log_file = f"metrics_{model_name}_{folder_type}.csv"

    df_results = run_pipeline(
        pdf_files, active_queries, active_gts,
        embedder, model_name, index,
        log_file=log_file
    )

    # ----------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"RESULTS SUMMARY — {model_name} | {folder_type.upper()}")
    print("=" * 60)

    for i, row in df_results.iterrows():
        print(f"\n🧾 Query {i+1}:")
        print(f"➡️  {active_queries[i]}")
        print("\n📌 Predicted Answer:")
        print(row["Answer"])
        print(f"✅ ROUGE-1: {row['ROUGE-1']:.3f} | ROUGE-2: {row['ROUGE-2']:.3f} | ROUGE-L: {row['ROUGE-L']:.3f}")
        print(f"✅ BLEU: {row['BLEU']:.3f} | METEOR: {row['METEOR']:.3f} | BERT-F1: {row['BERT-F1']:.3f}")
        print(f"✅ FCD: {row['FCD']:.3f} | Faithfulness: {row['Faithfulness']:.3f} | Bias: {row['Bias']:.3f}")
        print(f"✅ Terminology Precision: {row['Terminology_Precision']:.3f} | GT Coverage: {row['GroundTruth_Coverage(%)']:.2f}%")
        print(f"✅ Cosine Sim: {row['Cosine_Sim']:.3f} | TopK Accuracy: {row['TopK_Accuracy']:.2f}")
        print(f"🧩 Context Tokens: {row['Context_Tokens']} | Context Length: {row['Context_Length(words)']}")
        print(f"🖥️  CPU: {row['CPU_Usage(%)']}% | RAM: {row['RAM_Usage(GB)']} GB")
        print(f"⏱️  E2E Latency: {row['EndToEnd_Latency(sec)']:.2f}s | Throughput: {row['Throughput(q/s)']:.2f} q/s")

    again = input("\nRun another model on the same PDFs? (y/n): ").strip().lower()
    if again != "y":
        print("✅ Done.")
        break
