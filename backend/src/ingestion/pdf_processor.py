import fitz
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ============================================
# 1. Load NER Model
# ============================================
ner_model_name = "dslim/bert-base-NER"

ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

ner_pipeline = pipeline(
    "ner",
    model=ner_model,
    tokenizer=ner_tokenizer,
    aggregation_strategy="simple"
)

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
                        "page": page_num
                    })
                    buf = segment

        if buf:
            chunks.append({
                "text": buf.strip(),
                "source": pdf_path,
                "page": page_num
            })

    return chunks

# ============================================
# 3. NER-based Redaction (Names)
# ============================================
def run_ner_and_redact(text):
    entities = ner_pipeline(text)
    redacted_text = text

    for ent in entities:
        if ent['entity_group'] == 'PER':
            name = ent['word']
            redacted_text = re.sub(
                r'\b{}\b'.format(re.escape(name)),
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

    # Aadhaar (12 digits, with or without spaces)
    text = re.sub(r'\b\d{4}\s?\d{4}\s?\d{4}\b', '[REDACTED_AADHAAR]', text)

    # PAN (ABCDE1234F)
    text = re.sub(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', '[REDACTED_PAN]', text)

    # Email
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[REDACTED_EMAIL]', text)

    # Bank account (basic detection)
    text = re.sub(r'\b\d{9,18}\b', '[REDACTED_ACCOUNT]', text)

    return text

# ============================================
# 5. Categorization Logic
# ============================================
def categorize_text(text):
    """
    Categorizes text into categories (Criminal, Civil, Corporate) 
    and subcategories (Robbery, Fraud, etc.)
    """
    text_lower = text.lower()
    
    categories = {
        "Criminal": ["police", "arrest", "theft", "murder", "robbery", "fraud", "criminal", "ipc", "crpc"],
        "Civil": ["property", "divorce", "marriage", "contract", "landlord", "tenant", "civil"],
        "Corporate": ["company", "merger", "acquisition", "shares", "board", "director", "corporate"]
    }
    
    subcategories = {
        "Robbery": ["robbery", "theft", "snatching", "loot"],
        "Fraud": ["fraud", "cheating", "scam", "forgery"],
        "Property": ["land", "house", "building", "possession", "title"],
        "Marriage": ["marriage", "matrimonial", "husband", "wife", "alimony"]
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
# 6. Final Processing Pipeline
# ============================================
def process_text(text):
    text = run_ner_and_redact(text)
    text = redact_sensitive_info(text)
    return text