import os
import json
import pandas as pd

def run_ragas_eval(evaluation_data):
    """
    RAGAS Adapter.
    Expects evaluation_data as a list of dicts with:
    - query (str)
    - contexts (list of str)
    - answer (str)
    - ground_truth (str)
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        print("[*] Initializing RAGAS evaluation...")
        
        # Prepare dataset format for RAGAS
        data_dict = {
            "question": [item["query"] for item in evaluation_data],
            "contexts": [item["contexts"] for item in evaluation_data],
            "answer": [item["answer"] for item in evaluation_data],
            "ground_truth": [item["ground_truth"] for item in evaluation_data]
        }
        
        dataset = Dataset.from_dict(data_dict)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        print("[SUCCESS] RAGAS Evaluation Completed.")
        print(result)
        return result.to_pandas().to_dict(orient="records")
    except ImportError:
        print("\n[NOTE] RAGAS is not installed. To run official RAGAS metrics, run:")
        print("  pip install ragas datasets")
        return None

def run_deepeval_eval(evaluation_data):
    """
    DeepEval Adapter.
    """
    try:
        from deepeval import evaluate as deepeval_evaluate
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
        print("[*] Initializing DeepEval evaluation...")
        
        test_cases = []
        for item in evaluation_data:
            test_case = LLMTestCase(
                input=item["query"],
                actual_output=item["answer"],
                expected_output=item["ground_truth"],
                retrieval_context=item["contexts"]
            )
            test_cases.append(test_case)
            
        # Run evaluation
        faithfulness_metric = FaithfulnessMetric(threshold=0.5)
        answer_relevance_metric = AnswerRelevancyMetric(threshold=0.5)
        
        results = deepeval_evaluate(
            test_cases,
            metrics=[faithfulness_metric, answer_relevance_metric]
        )
        print("[SUCCESS] DeepEval Evaluation Completed.")
        return results
    except ImportError:
        print("\n[NOTE] DeepEval is not installed. To run official DeepEval metrics, run:")
        print("  pip install deepeval")
        return None

def export_to_standard_format(results_json_path, output_csv_path):
    """
    Utility to convert our internal benchmark JSON to standard datasets format
    used by RAGAS, DeepEval, or TruLens.
    """
    if not os.path.exists(results_json_path):
        print(f"[ERROR] Results file not found at {results_json_path}")
        return
        
    with open(results_json_path, "r") as f:
        data = json.load(f)
        
    # Standardize format
    formatted = []
    # If it's the dual pipeline comparative results
    if "primitive" in data and "enhanced" in data:
        print("[*] Formatting comparative results...")
        # Since comparative results contain averages, let's load individual query logs if available
    else:
        # Single pipeline run
        queries = data.get("queries", [])
        for q in queries:
            formatted.append({
                "query": q.get("query"),
                "contexts": q.get("retrieved_contexts", []),
                "answer": q.get("model_answer"),
                "ground_truth": q.get("ground_truth")
            })
            
    df = pd.DataFrame(formatted)
    df.to_csv(output_csv_path, index=False)
    print(f"[SUCCESS] Exported standardized evaluation dataset to {output_csv_path}")
