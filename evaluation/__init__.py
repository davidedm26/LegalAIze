"""Evaluation module for RAG system."""

from evaluation.data_loading import load_params, load_text, load_ground_truth_csv
from evaluation.metrics import (
    compute_mae,
    compute_note_similarity,
    compute_groundedness_score,
    compute_faithfulness_score,
)
from evaluation.preprocessing import (
    split_document_for_groundedness,
    build_requirement_question,
    select_relevant_contexts,
    extract_ground_truth_note,
)
from evaluation.mlflow_utils import log_case_input_artifacts
from evaluation.case_evaluation import evaluate_single_case
from evaluation.utils import slugify_case_name, normalize_case_selector, select_cases

__all__ = [
    "load_params",
    "load_text",
    "load_ground_truth_csv",
    "compute_mae",
    "compute_note_similarity",
    "compute_groundedness_score",
    "compute_faithfulness_score",
    "split_document_for_groundedness",
    "build_requirement_question",
    "select_relevant_contexts",
    "extract_ground_truth_note",
    "log_case_input_artifacts",
    "evaluate_single_case",
    "slugify_case_name",
    "normalize_case_selector",
    "select_cases",
]
