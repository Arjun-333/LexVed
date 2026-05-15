import fitz
import re
import spacy
from dotenv import load_dotenv

# Load variables from .env if present
load_dotenv()

# ============================================
# 1. Load SpaCy NER Model (High Speed CPU)
# ============================================
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Fallback if download failed in this specific environment
    import os
    os.system("python3 -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ============================================
# 2. Extract and Chunk PDF
# ============================================
def extract_chunks(pdf_path, chunk_size=200):
    doc = fitz.open(pdf_path)
    chunks = []

    citation_pattern = re.compile(
        r"(Section\s\d+[A-Za-z]*|Sec\.\s\d+[A-Za-z]*|\d+\s?Cr\.?\s?\d+)",
        re.IGNORECASE
    )

    for page_num, page in enumerate(doc):
        text = page.get_text("text")

        # Clean text
        text = re.sub(r"[*_]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Sentence splitting
        sentences = re.split(r'(?<=[.?!]) +|\n+', text)

        buf = ""
        for sent in sentences:
            parts = citation_pattern.split(sent)

            for part in parts:
                segment = part.strip()
                if not segment:
                    continue

                if len(buf.split()) + len(segment.split()) < chunk_size:
                    buf += " " + segment
                else:
                    chunks.append({
                        "text": buf.strip(),
                        "source": pdf_path,
                        "page": page_num + 1
                    })
                    buf = segment

        if buf:
            chunks.append({
                "text": buf.strip(),
                "source": pdf_path,
                "page": page_num + 1
            })

    return chunks

# ============================================
# 3. SpaCy-based Redaction (Fast)
# ============================================
def redact_names_spacy(text):
    doc = nlp(text)
    redacted_text = text
    # Reverse order to avoid index shifts if we were doing string manipulation, 
    # but here we use regex for safety with SpaCy words.
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            redacted_text = re.sub(
                r'\b{}\b'.format(re.escape(ent.text)),
                "[REDACTED_NAME]",
                redacted_text
            )
    return redacted_text

# ============================================
# 4. Regex-based Sensitive Info Redaction
# ============================================
def redact_sensitive_info(text):
    # Phone numbers (10-digit)
    text = re.sub(r'\b\d{10}\b', '[REDACTED_PHONE]', text)
    # Aadhaar (12 digits)
    text = re.sub(r'\b\d{4}\s?\d{4}\s?\d{4}\b', '[REDACTED_AADHAAR]', text)
    # PAN (ABCDE1234F)
    text = re.sub(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', '[REDACTED_PAN]', text)
    # Email
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[REDACTED_EMAIL]', text)
    return text

# ============================================
# 5. Categorization Logic
# ============================================
def categorize_text(text):
    text_lower = text.lower()
    categories = {
        "Criminal": ["police", "arrest", "theft", "murder", "robbery", "fraud", "criminal", "ipc", "crpc"],
        "Civil": ["property", "divorce", "marriage", "contract", "landlord", "tenant", "civil"],
    }
    subcategories = {
        "Robbery": ["robbery", "theft", "snatching", "loot"],
        "Fraud": ["fraud", "cheating", "scam", "forgery"],
        "Property": ["land", "house", "building", "possession", "title"],
    }
    
    detected_cat = "Uncategorized"
    detected_sub = "General"
    
    for cat, keywords in categories.items():
        if any(kw in text_lower for kw in keywords):
            detected_cat = cat
            break
    for sub, keywords in subcategories.items():
        if any(kw in text_lower for kw in keywords):
            detected_sub = sub
            break
    return detected_cat, detected_sub

# ============================================
# 6. Final Processing Pipeline (Turbo Batch)
# ============================================
def process_chunks_batch(chunks, batch_size=32):
    """
    Uses SpaCy's nlp.pipe for extreme speed on CPU.
    """
    texts = [c["text"] for c in chunks]
    processed_texts = []
    
    # SpaCy pipe is much faster with multi-processing
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=-1):
        text = doc.text
        # NER for names is now disabled (Keep names visible as per user request)
        # However, we still redact other sensitive info below
        redacted_text = text
        # Regex cleanup for phone, aadhaar, pan, email
        redacted_text = redact_sensitive_info(redacted_text)
        processed_texts.append(redacted_text)
            
    # Update chunks in place
    for i, chunk in enumerate(chunks):
        chunk["text"] = processed_texts[i]
        cat, sub = categorize_text(processed_texts[i])
        chunk["category"] = cat
        chunk["subcategory"] = sub
        
    return chunks

def process_text(text):
    """Legacy support."""
    text = redact_names_spacy(text)
    text = redact_sensitive_info(text)
    return text