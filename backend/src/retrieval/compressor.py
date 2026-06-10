import re

# Standard English stop words to filter out before matching
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def clean_tokens(text: str) -> set:
    """Tokenize and filter text into lowercase non-stop-word stems/words."""
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 1}

def compress_text(query: str, text: str, max_sentences: int = 3) -> str:
    """
    Compresses a single block of text by retaining only the top `max_sentences`
    sentences that contain the highest keyword match overlap with the query.
    
    Preserves original sentence order for readability.
    """
    if not text.strip():
        return ""

    # Split text into sentences (handles common abbreviations safely)
    sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = [s.strip() for s in sentence_end.split(text) if s.strip()]

    if len(sentences) <= max_sentences:
        return text

    query_tokens = clean_tokens(query)
    if not query_tokens:
        # Fallback if query contains only stop words
        return " ".join(sentences[:max_sentences])

    scored_sentences = []
    for idx, sentence in enumerate(sentences):
        sentence_tokens = clean_tokens(sentence)
        
        # Word overlap count
        overlap = len(query_tokens.intersection(sentence_tokens))
        
        # Priority bonus for statutory/legal identifiers (numbers, section labels)
        legal_terms = re.findall(r'\b(section|sec|art|article|rule|order|act|court|vs|versus|\d+)\b', sentence.lower())
        overlap += len(legal_terms) * 0.2
        
        # Length penalty to prevent long sentences from dominating
        length_penalty = len(sentence.split()) * 0.005
        score = max(0.0, overlap - length_penalty)
        
        scored_sentences.append((score, idx, sentence))

    # Sort by score descending, then by original index to keep top scores
    top_scored = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:max_sentences]
    
    # If no sentences matched any query terms, fallback to first few sentences
    if all(score == 0 for score, _, _ in top_scored):
        return " ".join(sentences[:max_sentences])

    # Re-sort by original index to preserve reading order
    top_ordered = sorted(top_scored, key=lambda x: x[1])
    
    return " ".join(sentence for _, _, sentence in top_ordered)
