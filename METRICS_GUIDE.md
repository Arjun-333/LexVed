# LexVed: Comprehensive Metrics Guide (M1 - M31)

This guide documents the 31 metrics evaluated by the LexVed pipeline and comparative audit benchmarks, detailing their purpose, mathematical formulations, and concrete examples.

---

## 1. Infrastructure Latency & Database Metrics (M1 - M3)

### M1: Embedding Latency (s)
* **Description:** The time taken by the embedding model (e.g., `multi-qa-mpnet-base-cos-v1`) to convert the input query text into a high-dimensional vector.
* **Formula:**
  $$\text{Latency} = T_{\text{end}} - T_{\text{start}}$$
* **Example:** Sending the query *"What is Section 138 of the NI Act?"* to the encoder starts at $T=0.000$ and returns the 768-dimension vector at $T=0.045\text{s}$. Latency = $0.045\text{s}$.

### M2: Index Size (Vectors)
* **Description:** The total count of document chunks currently indexed within the target vector database partition.
* **Example:** In a restored Pinecone database, querying index statistics returns `points_count = 19793`.

### M3: Retrieval Latency (s)
* **Description:** The time required to perform the search query against the vector database (and combine with sparse search in the enhanced hybrid pipeline).
* **Example:** Executing hybrid search + CrossEncoder reranking on a query takes $0.125\text{s}$.

---

## 2. Retrieval Quality (M4 - M5)

### M4: Cosine Similarity
* **Description:** Measures the cosine angle of the semantic vectors between the query and the retrieved document chunks.
* **Formula:**
  $$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
* **Example:** 
  * Query: *"cheque dishonour case"*
  * Retrieved Chunk: *"Under Section 138, if a cheque is dishonoured due to insufficient funds..."*
  * Vector similarity calculation yields $0.785$.

### M5: Recall@5
* **Description:** Measures whether the known relevant gold chunk text is present in the top 5 retrieved documents.
* **Formula:**
  $$\text{Recall@5} = \begin{cases} 1.0 & \text{if } \text{gold\_chunk} \in \text{top\_5\_retrieved} \\ 0.0 & \text{otherwise} \end{cases}$$
* **Example:**
  * Target Gold Chunk: *"Section 138 NI Act specifies a imprisonment term of up to two years."*
  * Top 5 Retrieved: includes the target chunk as the 3rd result.
  * Recall@5 = $1.0$.

---

## 3. Lexical and Semantic Similarity (M6 - M12)

### M6 - M8: ROUGE-1 / ROUGE-2 / ROUGE-L F1
* **Description:** 
  * **ROUGE-1:** Overlap of unigrams (single words).
  * **ROUGE-2:** Overlap of bigrams (two-word sequences).
  * **ROUGE-L:** Overlap based on the Longest Common Subsequence (LCS).
* **Formula:**
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **Example:**
  * Ground Truth: *"The petitioner is Balbir Kaur."*
  * Generated: *"The applicant is Balbir Kaur."*
  * Common unigrams: `The`, `is`, `Balbir`, `Kaur` (4 matches out of 5 words).
  * ROUGE-1 F1 $\approx 0.80$.

### M9: Context Words
* **Description:** The total count of words present in the retrieved context passed to the generation LLM.
* **Example:** Slicing the top 5 chunks yields $1,250$ words.

### M10: BLEU Score
* **Description:** Measures n-gram precision of the generated text against the ground truth answer with a brevity penalty.
* **Example:** If the model repeats correct words but has poor grammar or incorrect order, BLEU penalizes the sequence matching, yielding a low score (e.g. $0.35$).

### M11: METEOR Score
* **Description:** Evaluates translation/generation quality by matching exact words, stemmed forms, and synonyms.
* **Example:** Match "agree" in ground truth to "concur" in model answer using WordNet synonym matches.

### M12: BERTScore F1
* **Description:** Computes semantic similarity by matching token embeddings of the generated and reference sentences using cosine similarity.
* **Example:**
  * Ground Truth: *"The court dismissed the application."*
  * Generated Answer: *"The judge rejected the motion."*
  * Though word overlap is low (different vocabulary), BERTScore F1 is high ($\approx 0.91$) due to matching contextual token representations.

---

## 4. Factual Grounding & Faithfulness (M13 - M15)

### M13: Factual Consistency Deviation (FCD)
* **Description:** The complementary inverse of faithfulness. High deviation indicates the presence of unsupported or hallucinated statements.
* **Formula:**
  $$\text{FCD} = 1.0 - \text{Faithfulness (M14)}$$
* **Example:** If Faithfulness is $0.85$, FCD is $0.15$.

### M14: Faithfulness
* **Description:** The ratio of statements in the generated response that are directly supported by the retrieved context.
* **Formula:**
  $$\text{Faithfulness} = \frac{\text{Number of supported statements}}{\text{Total statements generated}}$$
* **Example:**
  * Model generates 3 claims: (1) *"A cheque was written on Oct 5"*, (2) *"The cheque was returned unpaid"*, (3) *"The penalty is 10,000 rupees"*.
  * Chunks support claims (1) and (2). Claim (3) is not in the context.
  * Faithfulness = $2 / 3 \approx 0.667$.

### M15: Ground Truth Coverage (%)
* **Description:** The percentage of stemmed words from the Ground Truth that are found in the retrieved context documents.
* **Formula:**
  $$\text{Coverage} = \frac{|\text{Stemmed GT Tokens} \cap \text{Stemmed Context Tokens}|}{|\text{Stemmed GT Tokens}|} \times 100\%$$
* **Example:**
  * Ground Truth: *"The judge acquitted the accused person."*
    * Stemmed tokens: `judg`, `acquit`, `accus`, `person`
  * Context: *"The court decided to acquit the accused."*
    * Stemmed tokens: `court`, `decid`, `acquit`, `accus`
  * Intersection: `acquit`, `accus` (2 matches out of 4).
  * Coverage = $2 / 4 = 50\%$.

---

## 5. System Runtime & Efficiency (M16 - M19)

### M16: End-to-End Latency (s)
* **Description:** The elapsed time from query receipt to the completion of model response generation.
* **Formula:**
  $$\text{E2E Latency} = \text{Retrieval Latency} + \text{Generation Latency}$$

### M17: Throughput (QPS)
* **Description:** Queries processed per second.
* **Formula:**
  $$\text{QPS} = \frac{1}{\text{E2E Latency}}$$

### M18 & M19: CPU & RAM Utilization
* **Description:** System resource metrics captured during execution using `psutil`.

---

## 6. Legal Compliance & Reasoning KPIs (M20 - M24)

### M20: Citation Accuracy (%)
* **Description:** An objective check of whether the statutory and case citations present in the Ground Truth are cited in the generated answer.
* **Formula:**
  $$\text{Citation Accuracy} = \frac{|\text{Citations in Generated} \cap \text{Citations in Ground Truth}|}{|\text{Citations in Ground Truth}|} \times 100\%$$
* **Example:**
  * Ground Truth: *"Violates Section 138 and Section 141 of the NI Act."*
  * Model Answer: *"Violates Section 138 of the NI Act."*
  * Citation Accuracy = $1 / 2 = 50\%$.

### M21: Terminology Precision
* **Description:** Checks if critical legal terms (e.g. `writ petition`, `habeas corpus`) are translated and applied accurately.

### M22 - M24: Precedent Match, Regulatory Alignment, & Bias Score
* **Description:** Automated LLM-judged criteria verifying legal compliance constraints, bias minimization, and preceding ruling alignment.

---

## 7. LLM Stream and Generation Performance (M25 - M27)

### M25: Prefill Latency (s)
* **Description:** The time taken by the LLM inference engine to process the initial prompt and retrieval context before generating text.
* **Example:** Extracted from Groq's usage header `prompt_time` $\approx 0.182\text{s}$.

### M26: Time to First Token (TTFT) (s)
* **Description:** The elapsed time between initiation of the LLM API request and the receipt of the very first generated text chunk token.
* **Example:** TTFT = $0.235\text{s}$.

### M27: Generation Throughput (tokens/sec)
* **Description:** The speed at which text tokens are generated.
* **Formula:**
  $$\text{Throughput} = \frac{\text{Total Generated Tokens}}{\text{Generation Latency} - \text{TTFT}}$$

---

## 8. Standard Information Retrieval Benchmarks (M28 - M31)

### M28: Recall@10
* **Description:** Measures if the known target gold document chunk is retrieved in the top 10 results.
* **Formula:**
  $$\text{Recall@10} = \begin{cases} 1.0 & \text{if } \text{gold\_chunk} \in \text{top\_10\_retrieved} \\ 0.0 & \text{otherwise} \end{cases}$$

### M29: Mean Reciprocal Rank (MRR)
* **Description:** Reciprocal of the rank where the relevant gold chunk appears (calculated up to rank 10).
* **Formula:**
  $$\text{MRR} = \frac{1}{\text{Rank of Gold Chunk}}$$
* **Example:**
  * If the gold chunk is found at Rank 1, MRR = $1.0$.
  * If found at Rank 4, MRR = $0.25$.
  * If not in the top 10, MRR = $0.0$.

### M30: nDCG@10
* **Description:** Normalized Discounted Cumulative Gain at rank 10. Rewards the system for retrieving the target chunk at higher ranks.
* **Formula:**
  $$\text{nDCG@10} = \frac{1}{\log_2(\text{Rank} + 1)}$$
* **Example:**
  * If the gold chunk is at Rank 1 (index 0), $\text{nDCG} = 1 / \log_2(2) = 1.0$.
  * If the gold chunk is at Rank 3 (index 2), $\text{nDCG} = 1 / \log_2(4) = 0.50$.
  * If not in top 10, $\text{nDCG} = 0.0$.

### M31: Precision@5
* **Description:** The fraction of retrieved documents in the top 5 that are relevant. Since there is exactly one target gold chunk per query:
* **Formula:**
  $$\text{Precision@5} = \frac{\text{Relevant Chunks in Top 5}}{5} = \begin{cases} 0.20 & \text{if } \text{gold\_chunk} \in \text{top\_5\_retrieved} \\ 0.0 & \text{otherwise} \end{cases}$$
