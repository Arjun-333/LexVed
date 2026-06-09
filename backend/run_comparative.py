"""
Comparative Benchmarking — runs the same evaluation queries across all embedding models.
Produces a side-by-side comparison in comparative_results.json.

References the Enhanced Pipeline methodology from LexVed_Institutional_Audit.ipynb.
"""
import os
import json
import time
import sys
from src.utils.config_manager import load_config, set_active_model, get_active_model_name
from src.retrieval.retriever import invalidate_bm25
from run_metrics import run_evaluation

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COMPARATIVE_PATH = os.path.join(PROJECT_ROOT, "comparative_results.json")
INTERMEDIATE_PATH = os.path.join(PROJECT_ROOT, "intermediate_results.json")
EVAL_RESULTS_PATH = os.path.join(PROJECT_ROOT, "evaluation_results.json")


def run_comparative():
    config = load_config()
    all_models = list(config.get("models", {}).keys())
    original_model = get_active_model_name()

    resume_flag = "--resume" in sys.argv

    results = {}
    if resume_flag and os.path.exists(INTERMEDIATE_PATH):
        try:
            with open(INTERMEDIATE_PATH, "r") as f:
                results = json.load(f)
            print(f"[LexVed] Resuming from previously saved results for models: {list(results.keys())}")
        except Exception as e:
            print(f"[LexVed] Could not resume intermediate results: {e}")

    total = len(all_models)

    print(f"\n{'='*80}")
    print(f" LEXVED COMPARATIVE BENCHMARK -- {total} Models")
    print(f"{'='*80}")

    for idx, model_name in enumerate(all_models):
        if resume_flag and model_name in results and "error" not in results[model_name]:
            print(f"[{idx+1}/{total}] Skipping already completed model: {model_name}")
            continue

        print(f"\n[{idx+1}/{total}] Benchmarking: {model_name}")

        # Update progress file (preserve PID)
        pid = None
        if os.path.exists(COMPARATIVE_PATH):
            try:
                with open(COMPARATIVE_PATH, "r") as f:
                    old = json.load(f)
                    pid = old.get("pid")
            except Exception:
                pass

        with open(COMPARATIVE_PATH, "w") as f:
            json.dump({
                "status": "processing",
                "progress": f"Benchmarking model {idx+1}/{total}: {model_name}",
                "completed_models": list(results.keys()),
                "current_model": model_name,
                "pid": pid or os.getpid()
            }, f, indent=2)

        # Switch to this model & invalidate caches
        set_active_model(model_name)
        invalidate_bm25()

        # Run evaluation (writes to evaluation_results.json)
        try:
            run_evaluation()

            # Read the results
            if os.path.exists(EVAL_RESULTS_PATH):
                with open(EVAL_RESULTS_PATH, "r") as f:
                    eval_data = json.load(f)
                results[model_name] = {
                    "summary": eval_data.get("summary", {}),
                    "details": eval_data.get("details", []),
                    "system_info": eval_data.get("system_info", {})
                }
                with open(INTERMEDIATE_PATH, "w") as f:
                    json.dump(results, f, indent=4)
            else:
                results[model_name] = {"error": "No evaluation results generated"}

        except Exception as e:
            print(f"  Error benchmarking {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    # Restore original model
    set_active_model(original_model)
    invalidate_bm25()

    # Build comparative summary
    metric_keys = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10",
                   "M11", "M12", "M13", "M14", "M15", "M16", "M17", "M18", "M19",
                   "M20", "M21", "M22", "M23", "M24", "M25", "M26", "M27"]

    # Lower-is-better metrics
    lower_is_better = {"M1", "M3", "M13", "M16", "M18", "M19", "M24", "M25", "M26"}

    comparison_table = {}
    for mk in metric_keys:
        comparison_table[mk] = {}
        for model_name in all_models:
            if model_name in results and "summary" in results[model_name]:
                comparison_table[mk][model_name] = results[model_name]["summary"].get(mk, None)
            else:
                comparison_table[mk][model_name] = None

    # Find best model per metric
    best_per_metric = {}
    for mk in metric_keys:
        best_model = None
        best_val = None
        for model_name, val in comparison_table[mk].items():
            if val is not None:
                try:
                    f_val = float(val)
                    if mk in lower_is_better:
                        if best_val is None or f_val < best_val:
                            best_val = f_val
                            best_model = model_name
                    else:
                        if best_val is None or f_val > best_val:
                            best_val = f_val
                            best_model = model_name
                except (ValueError, TypeError):
                    pass
        best_per_metric[mk] = best_model

    final_report = {
        "status": "complete",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_benchmarked": all_models,
        "comparison_table": comparison_table,
        "best_per_metric": best_per_metric,
        "detailed_results": results,
        "progress": f"Comparative benchmark complete -- {len(all_models)} models evaluated"
    }

    with open(COMPARATIVE_PATH, "w") as f:
        json.dump(final_report, f, indent=4)

    # Clean up intermediate file
    if os.path.exists(INTERMEDIATE_PATH):
        try:
            os.remove(INTERMEDIATE_PATH)
        except Exception:
            pass

    print(f"\n[SUCCESS] Comparative results saved for {len(all_models)} models.")


if __name__ == "__main__":
    run_comparative()
