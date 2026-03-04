"""Evaluation module for RAG system."""
# This module allows the use of functions as a library.

from evaluation.data_loading import load_params, load_text, load_ground_truth_csv
from evaluation.metrics import (
    compute_mae,
    compute_subrequirements_ragas_metrics,
)

from evaluation.mlflow_utils import log_case_input_artifacts
from evaluation.case_evaluation import evaluate_single_case
from evaluation.utils import slugify_case_name, normalize_case_selector, select_cases

__all__ = [
    "load_params",
    "load_text",
    "load_ground_truth_csv",
    "compute_mae",
    "compute_subrequirements_ragas_metrics",
    "log_case_input_artifacts",
    "evaluate_single_case",
    "slugify_case_name",
    "normalize_case_selector",
    "select_cases",
]
