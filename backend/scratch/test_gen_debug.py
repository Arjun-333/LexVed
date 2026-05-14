from src.generation.generator import generate_answer
import os

print("Testing LexVed Generator...")
ans, _, _ = generate_answer("What is the capital of France?", "The capital of France is Paris.")
print(f"Generated Answer: {ans}")
