"""
Comparative Benchmarking — runs the same evaluation queries across all embedding models.
Produces a side-by-side comparison in comparative_results.json.
"""
import os
import json
import time
from src.utils.config_manager import load_config, set_active_model, get_active_model_name
from run_metrics import run_evaluation

def run_comparative():
    config = load_config()
    all_models = list(config.get("models", {}).keys())
    original_model = get_active_model_name()
    
    results = {}
    total = len(all_models)
    
    print(f"\n{'='*80}")
    print(f" LEXVED COMPARATIVE BENCHMARK — {total} Models")
    print(f"{'='*80}")
    
    for idx, model_name in enumerate(all_models):
        print(f"\n[{idx+1}/{total}] Benchmarking: {model_name}")
        
        # Update progress file
        with open("comparative_results.json", "w") as f:
            json.dump({
                "status": "processing",
                "progress": f"Benchmarking model {idx+1}/{total}: {model_name}",
                "completed_models": list(results.keys()),
                "current_model": model_name
            }, f, indent=2)
        
        # Switch to this model
        set_active_model(model_name)
        
        # Run evaluation (this updates evaluation_results.json)
        try:
            run_evaluation()
            
            # Read the results
            if os.path.exists("evaluation_results.json"):
                with open("evaluation_results.json", "r") as f:
                    eval_data = json.load(f)
                results[model_name] = {
                    "summary": eval_data.get("summary", {}),
                    "details": eval_data.get("details", []),
                    "system_info": eval_data.get("system_info", {})
                }
            else:
                results[model_name] = {"error": "No evaluation results generated"}
                
        except Exception as e:
            print(f"  Error benchmarking {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    # Restore original model
    set_active_model(original_model)
    
    # Build comparative summary
    metric_keys = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10",
                   "M11", "M12", "M13", "M14", "M15", "M16", "M17", "M18", "M19",
                   "M20", "M21", "M22", "M23", "M24"]
    
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
                # For M13 (hallucination rate) and M17/M18/M19 (latency/cost), lower is better
                if mk in ["M13", "M16", "M17", "M18", "M19", "M1", "M3"]:
                    if best_val is None or val < best_val:
                        best_val = val
                        best_model = model_name
                else:
                    if best_val is None or val > best_val:
                        best_val = val
                        best_model = model_name
        best_per_metric[mk] = best_model
    
    final_report = {
        "status": "complete",
        "timestamp": time.ctime(),
        "models_benchmarked": all_models,
        "comparison_table": comparison_table,
        "best_per_metric": best_per_metric,
        "detailed_results": results,
        "progress": f"Comparative benchmark complete — {len(all_models)} models evaluated"
    }
    
    with open("comparative_results.json", "w") as f:
        json.dump(final_report, f, indent=4)
    
    print(f"\n[SUCCESS] Comparative results saved for {len(all_models)} models.")

if __name__ == "__main__":
    run_comparative()
