import os
import json
import yaml
import csv
import random
from typing import Dict, List, Any

import requests
from dotenv import load_dotenv

import mlflow

load_dotenv() # Load environment variables from .env file if present

# Configure MLflow for DagsHub
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")

def load_params() -> Dict[str, Any]:
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_ground_truth_csv(path: str) -> Dict[str, Dict[str, Any]]:
    """Load ground truth CSV into a dict keyed by Mapped_ID.
    """
    gt: Dict[str, Dict[str, Any]] = {} # Result dict

    for encoding in ("utf-8-sig", "latin-1"): # Try multiple encodings
        try:
            with open(path, "r", encoding=encoding) as f:
                reader = csv.DictReader(f) 
                for row in reader: 
                    # Use mapped_id as key
                    mapped_id = row.get("Mapped_ID") or row.get("Mapped ID") 
                    if not mapped_id:
                        continue
                    gt[mapped_id] = row 
            return gt
        except UnicodeDecodeError:
            continue

    # If all encodings fail, re-raise
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode CSV file: {path}")


def call_audit(backend_url: str, document_text: str, timeout: int = 60) -> List[Dict[str, Any]]:
    url = backend_url.rstrip("/") + "/audit"
    resp = requests.post(url, json={"document_text": document_text}, timeout=timeout)
    if (not resp.ok):
        print(f"⚠ Audit request failed: {resp.status_code} - {resp.text}")
        resp.raise_for_status()
    data = resp.json()
    
    return data.get("requirements", []) # List of requirement dicts


# Distance metrics
def compute_mae(gt_scores: List[float], pred_scores: List[float]) -> float:
    if not gt_scores:
        return 0.0
    diffs = [abs(g - p) for g, p in zip(gt_scores, pred_scores)] 
    return sum(diffs) / len(diffs)


def evaluate_single_case(gt_path: str, doc_path: str, backend_url: str) -> Dict[str, Any]:
    """Evaluate one ground-truth/report pair against the /audit endpoint."""
    ground_truth = load_ground_truth_csv(gt_path) # Load GT as dict keyed by Mapped_ID
    document_text = load_text(doc_path) # Load document text to audit

    predictions = call_audit(backend_url, document_text) # Call audit endpoint to get predictions

    gt_scores: List[float] = [] # Ground truth scores
    pred_scores: List[float] = [] # Predicted scores

    for pred in predictions: # Each predicted requirement report
        mapped_id = pred.get("Mapped_ID")
        if not mapped_id or mapped_id not in ground_truth:
            continue
        gt_row = ground_truth[mapped_id] # Corresponding GT row for this Mapped_ID
        try:
            gt_score = float(gt_row.get("Score", "0")) # GT score from CSV, default to 0 if missing or invalid
        except ValueError: # (N/A) case is included here
            continue

        try:
            pred_score = float(pred.get("Score")) # Predicted score from audit response
        except (TypeError, ValueError):
            continue

        gt_scores.append(gt_score)
        pred_scores.append(pred_score)

    mae = compute_mae(gt_scores, pred_scores) # Compute MAE for this particular case

    return {
        "num_pairs": len(gt_scores),
        "mae_score": mae,
    }


def main() -> None:
    params = load_params()
    eval_params = params.get("evaluation", {})
    precompute_params = params.get("precompute", {})
    ingestion_params = params.get("ingestion", {})


    backend_url = os.getenv("BACKEND_URL") 
    random_seed = int(eval_params.get("random_seed", 42))
    
    llm_model = eval_params.get("llm_model")
    llm_temperature = float(eval_params.get("llm_temperature"))
    metrics_output = eval_params.get("metrics_output", "metrics/rag_eval.json")
    gt_cases = eval_params.get("ground_truth", [])

    # Set seed for reproducibility where possible
    random.seed(random_seed)

    # Prepare metrics dir
    metrics_dir = os.path.dirname(metrics_output)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    # Aggregate metrics across all cases
    all_results: List[Dict[str, Any]] = []

    # Optional: configure MLflow (env first, then params)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = eval_params.get("mlflow_experiment", "rag_evaluation")

    if mlflow is not None and tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    run_ctx = (
        mlflow.start_run(run_name="rag_eval")
        if mlflow is not None and tracking_uri
        else None
    )
    
    mapping_path = ingestion_params.get(
        "mapping_path",
        os.path.join(ingestion_params.get("raw_data_dir", "data"), "mapping.json"),
    )

    if run_ctx is not None:
        print(f"✓ MLflow run started: {run_ctx.info.run_id}")
    else: 
        print("⚠ MLflow tracking URI not configured. Metrics will not be logged to MLflow.")
        

    try:
        # Log static params to MLflow if available
        if run_ctx is not None:
            mlflow.log_param("backend_url", backend_url)
            mlflow.log_param("random_seed", random_seed)
            mlflow.log_param("llm_model", llm_model)
            mlflow.log_param("llm_temperature", llm_temperature)
            mlflow.log_param("precompute_top_k", precompute_params.get("top_k", 3))
            mlflow.log_param("chunk_size", ingestion_params.get("chunk_size"))
            mlflow.log_param("chunk_overlap", ingestion_params.get("chunk_overlap"))
        

        for case in gt_cases:
            name = case.get("name", "unknown") # Evaluation case name for logging
            doc_path = case["document_path"] # Path to the document for this case
            report_path = case["report_path"] # Path to the ground truth report for this case

            print(f"Evaluating case: {name}")
            if not os.path.exists(doc_path):
                print(f"⚠ Document not found: {doc_path}")
                continue # Skip this case if document is missing
            if not os.path.exists(report_path):
                print(f"⚠ Ground truth report not found: {report_path}")
                continue # Skip this case if ground truth report is missing

            res = evaluate_single_case(report_path, doc_path, backend_url) # Evaluate this evaluation case
            res["name"] = name # Add case name to results
            all_results.append(res) # Append to all results

            if run_ctx is not None:
                # Log per-case metrics under a prefix
                mlflow.log_metric(f"{name}_mae_score", res["mae_score"])
                mlflow.log_metric(f"{name}_num_pairs", res["num_pairs"])

        # Compute global aggregates
        if all_results:
            total_pairs = sum(r["num_pairs"] for r in all_results)
            # Weighted MAE by number of pairs
            weighted_mae = (
                sum(r["mae_score"] * r["num_pairs"] for r in all_results) / total_pairs
                if total_pairs > 0
                else 0.0
            )
        else:
            total_pairs = 0
            weighted_mae = 0.0

        summary = {
            "total_cases": len(all_results),
            "total_pairs": total_pairs,
            "weighted_mae_score": weighted_mae,
            "cases": all_results,
            "config_snapshot": {
                "chunk_size": ingestion_params.get("chunk_size"),
                "chunk_overlap": ingestion_params.get("chunk_overlap"),
            },
        }

        # Save metrics to JSON (for DVC)
        with open(metrics_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if run_ctx is not None:
            mlflow.log_metric("weighted_mae_score", weighted_mae)
            mlflow.log_metric("total_pairs", total_pairs)
            # Log the metrics file as artifact
            mlflow.log_artifact(metrics_output)
            if os.path.exists(mapping_path):
                mlflow.log_artifact(mapping_path, artifact_path="inputs")

        print("✓ RAG evaluation completed.")

    finally:
        if run_ctx is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
