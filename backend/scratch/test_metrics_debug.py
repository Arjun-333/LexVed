import evaluate
import os

try:
    print("Loading BERTScore...")
    metric_bertscore = evaluate.load("bertscore")
    print("Computing BERTScore...")
    res = metric_bertscore.compute(predictions=["The cat is on the mat."], references=["A cat is on a mat."], lang="en")
    print(f"Result: {res}")
except Exception as e:
    print(f"Error: {e}")
