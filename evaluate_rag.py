import os
import json
import yaml
import csv
import random
from typing import Dict, List, Any

import requests
from dotenv import load_dotenv

import mlflow


# Load environment variables from .env (for MLFLOW_TRACKING_URI, BACKEND_URL, etc.)
load_dotenv()


def load_params() -> Dict[str, Any]:
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_ground_truth_csv(path: str) -> Dict[str, Dict[str, Any]]:
    """Load ground truth CSV into a dict keyed by Mapped_ID.

    Tries UTF-8 first, then falls back to latin-1 to handle Windows-encoded files.
    """
    gt: Dict[str, Dict[str, Any]] = {}

    # Try a couple of common encodings
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mapped_id = row.get("Mapped_ID") or row.get("Mapped ID")
                    if not mapped_id:
                        continue
                    gt[mapped_id] = row
            return gt
        except UnicodeDecodeError:
            continue

    # If all encodings fail, re-raise for visibility
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode CSV file: {path}")


def call_audit(backend_url: str, document_text: str, timeout: int = 60) -> List[Dict[str, Any]]:
    url = backend_url.rstrip("/") + "/audit"
    resp = requests.post(url, json={"document_text": document_text}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # data["requirements"] is a list of objects with Mapped_ID, Requirement_Name, Score, Auditor_Notes
    return data.get("requirements", [])


def compute_mae(gt_scores: List[float], pred_scores: List[float]) -> float:
    if not gt_scores:
        return 0.0
    diffs = [abs(g - p) for g, p in zip(gt_scores, pred_scores)]
    return sum(diffs) / len(diffs)


def evaluate_single_case(gt_path: str, doc_path: str, backend_url: str) -> Dict[str, Any]:
    """Evaluate one ground-truth/report pair against the /audit endpoint."""
    ground_truth = load_ground_truth_csv(gt_path)
    document_text = load_text(doc_path)

    predictions = call_audit(backend_url, document_text)

    gt_scores: List[float] = []
    pred_scores: List[float] = []

    for pred in predictions:
        mapped_id = pred.get("Mapped_ID")
        if not mapped_id or mapped_id not in ground_truth:
            continue
        gt_row = ground_truth[mapped_id]
        try:
            gt_score = float(gt_row.get("Score (0-5)", "0"))
        except ValueError:
            continue

        # Our model currently outputs Score 1-5; we keep it as-is for now.
        try:
            pred_score = float(pred.get("Score"))
        except (TypeError, ValueError):
            continue

        gt_scores.append(gt_score)
        pred_scores.append(pred_score)

    mae = compute_mae(gt_scores, pred_scores)

    return {
        "num_pairs": len(gt_scores),
        "mae_score": mae,
    }


def main() -> None:
    params = load_params()
    eval_params = params.get("evaluation", {})
    precompute_params = params.get("precompute", {})

    # Prefer BACKEND_URL from environment, fallback to params
    backend_url = os.getenv("BACKEND_URL") or eval_params.get("backend_url", "http://localhost:8000")
    random_seed = int(eval_params.get("random_seed", 42))
    llm_model = eval_params.get("llm_model", "gpt-3.5-turbo")
    llm_temperature = float(eval_params.get("llm_temperature", 0.1))
    metrics_output = eval_params.get("metrics_output", "metrics/rag_eval.json")
    gt_cases = eval_params.get("ground_truth", [])

    # Set seed for reproducibility where possible (not for the remote LLM itself)
    random.seed(random_seed)

    # Prepare metrics dir
    metrics_dir = os.path.dirname(metrics_output)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    # Aggregate metrics across all cases
    all_results: List[Dict[str, Any]] = []

    # Optional: configure MLflow (env first, then params)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or eval_params.get("mlflow_tracking_uri")
    experiment_name = eval_params.get("mlflow_experiment", "rag_evaluation")

    if mlflow is not None and tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    run_ctx = (
        mlflow.start_run(run_name="rag_eval")
        if mlflow is not None and tracking_uri
        else None
    )

    try:
        # Log static params to MLflow if available
        if run_ctx is not None:
            mlflow.log_param("backend_url", backend_url)
            mlflow.log_param("random_seed", random_seed)
            mlflow.log_param("llm_model", llm_model)
            mlflow.log_param("llm_temperature", llm_temperature)
            mlflow.log_param("precompute_top_k", precompute_params.get("top_k", 3))

        for case in gt_cases:
            name = case.get("name", "unknown")
            doc_path = case["document_path"]
            report_path = case["report_path"]

            print(f"Evaluating case: {name}")
            if not os.path.exists(doc_path):
                print(f"⚠ Document not found: {doc_path}")
                continue
            if not os.path.exists(report_path):
                print(f"⚠ Ground truth report not found: {report_path}")
                continue

            res = evaluate_single_case(report_path, doc_path, backend_url)
            res["name"] = name
            all_results.append(res)

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
        }

        # Save metrics to JSON (for DVC)
        with open(metrics_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if run_ctx is not None:
            mlflow.log_metric("weighted_mae_score", weighted_mae)
            mlflow.log_metric("total_pairs", total_pairs)
            # Log the metrics file as artifact
            mlflow.log_artifact(metrics_output)

        print("✓ RAG evaluation completed.")

    finally:
        if run_ctx is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
