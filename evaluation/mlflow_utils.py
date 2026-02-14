"""MLflow utilities for logging evaluation artifacts."""

import os
import json
import tempfile
import mlflow

from evaluation.data_loading import load_ground_truth_csv
from evaluation.utils import slugify_case_name


def log_case_input_artifacts(case_name: str, doc_path: str, report_path: str) -> None:
    """Upload document and ground truth report as MLflow artifacts."""
    if mlflow is None or mlflow.active_run() is None:
        return
    artifact_prefix = f"cases/{slugify_case_name(case_name)}/inputs"
    if os.path.exists(doc_path):
        mlflow.log_artifact(doc_path, artifact_path=artifact_prefix)
    if os.path.exists(report_path):
        # Load and convert CSV report to JSON format
        report_data = load_ground_truth_csv(report_path)
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as tmp_json:
            json.dump(report_data, tmp_json, ensure_ascii=False, indent=2)
            tmp_json_path = tmp_json.name
        # Rename to report.json before logging
        report_json_path = os.path.join(os.path.dirname(tmp_json_path), "report.json")
        os.rename(tmp_json_path, report_json_path)
        mlflow.log_artifact(report_json_path, artifact_path=artifact_prefix)
        os.remove(report_json_path)
