import os
import json
import yaml
import csv
import random
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

import mlflow

try:
    from datasets import Dataset
except ImportError:
    Dataset = None  # type: ignore

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import Groundedness
except ImportError:
    ragas_evaluate = None  # type: ignore
    Groundedness = None  # type: ignore

RAGAS_AVAILABLE = Dataset is not None and ragas_evaluate is not None and Groundedness is not None

try:
    from backend import rag_engine
except ImportError:
    rag_engine = None  # type: ignore

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


# Distance metrics
def compute_mae(gt_scores: List[float], pred_scores: List[float]) -> float:
    if not gt_scores:
        return 0.0
    diffs = [abs(g - p) for g, p in zip(gt_scores, pred_scores)] 
    return sum(diffs) / len(diffs)


def compute_note_similarity(gt_note: str, pred_note: str) -> float:
    """Compute cosine similarity between ground-truth and predicted notes using embeddings."""
    if not gt_note or not pred_note:
        return 0.0
    if rag_engine is None or rag_engine.embedding_model is None:
        raise RuntimeError("Sentence embedding model not available. Ensure rag_engine.init_rag() has run.")

    embeddings = rag_engine.embedding_model.encode(
        [gt_note, pred_note],
        convert_to_numpy=True,
    )
    gt_vec, pred_vec = embeddings
    gt_norm = np.linalg.norm(gt_vec)
    pred_norm = np.linalg.norm(pred_vec)
    if not gt_norm or not pred_norm:
        return 0.0
    similarity = float(np.dot(gt_vec, pred_vec) / (gt_norm * pred_norm))
    # Numerical noise can push slightly outside [-1,1]
    return max(min(similarity, 1.0), -1.0)


def slugify_case_name(name: str) -> str:
    base = name or "case"
    slug = re.sub(r"[^a-z0-9_-]+", "-", base.lower()).strip("-")
    return slug or "case"


def normalize_case_selector(selector: Any) -> Optional[List[int]]:
    if selector is None:
        return None
    if isinstance(selector, int):
        return [selector]
    if isinstance(selector, list):
        normalized: List[int] = []
        for item in selector:
            try:
                normalized.append(int(item))
            except (TypeError, ValueError):
                print(f"⚠ Ignoring invalid case selector entry: {item}")
        return normalized or None
    try:
        value = int(selector)
        return [value]
    except (TypeError, ValueError):
        print(f"⚠ Unsupported case selector type: {selector}")
        return None


def select_cases(gt_cases: List[Dict[str, Any]], selector: Optional[List[int]]) -> List[Dict[str, Any]]:
    if not selector:
        return gt_cases
    selected: List[Dict[str, Any]] = []
    total = len(gt_cases)
    for idx in selector:
        if idx < 1 or idx > total:
            print(f"⚠ Case index {idx} is out of range (1-{total}). Skipping.")
            continue
        selected.append(gt_cases[idx - 1])
    return selected


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


def split_document_for_groundedness(
    document_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    if not document_text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(document_text)


def build_requirement_question(mapped_id: Optional[str], requirement_name: Optional[str]) -> str:
    identifier = mapped_id or "Unknown requirement"
    title = requirement_name or identifier
    mapping_data = getattr(rag_engine, "mapping", None) or {}
    req_metadata = mapping_data.get(requirement_name or "", {}) if mapping_data else {}
    iso_ref = req_metadata.get("iso_ref")
    iso_text = req_metadata.get("iso_control_text")
    suffix_parts = [part for part in (iso_ref, iso_text) if part]
    if suffix_parts:
        suffix = " | ".join(suffix_parts)
        return f"{title} ({identifier}) — {suffix}"
    return f"{title} ({identifier})"


def select_relevant_contexts(
    reference_text: str,
    chunk_texts: List[str],
    chunk_embeddings: Optional[np.ndarray],
    chunk_norms: Optional[np.ndarray],
    top_k: int,
) -> List[str]:
    if not chunk_texts:
        return []
    if top_k <= 0:
        return []
    if (
        not reference_text
        or chunk_embeddings is None
        or chunk_norms is None
        or rag_engine is None
        or rag_engine.embedding_model is None
    ):
        return chunk_texts[:top_k]

    note_vec = rag_engine.embedding_model.encode(
        [reference_text],
        convert_to_numpy=True,
    )[0]
    note_norm = np.linalg.norm(note_vec)
    if not note_norm:
        return chunk_texts[:top_k]

    denom = (chunk_norms * note_norm) + 1e-8
    similarities = np.dot(chunk_embeddings, note_vec) / denom
    ranked_indices = np.argsort(similarities)[::-1]
    selected: List[str] = []
    for idx in ranked_indices[:top_k]:
        selected.append(chunk_texts[idx])
    return selected or chunk_texts[:top_k]


def extract_ground_truth_note(row: Dict[str, Any]) -> Optional[str]:
    return (
        row.get("Auditor Notes")
        or row.get("auditor_notes")
        or row.get("Auditor_Notes")
    )

def evaluate_single_case(
    *,
    case_name: str,
    gt_path: str,
    doc_path: str,
    chunk_size: int,
    chunk_overlap: int,
    groundedness_top_k: int,
    case_artifact_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate one ground-truth/report pair using the local RAG engine."""
    ground_truth = load_ground_truth_csv(gt_path) # Load GT as dict keyed by Mapped_ID
    document_text = load_text(doc_path) # Load document text to audit

    if rag_engine is None:
        raise RuntimeError("backend.rag_engine is not available. Run evaluate_rag from the project root.")

    audit_response = rag_engine.audit_document(document_text)
    predictions = [report.model_dump() for report in audit_response.requirements]

    document_chunks = split_document_for_groundedness(
        document_text,
        chunk_size,
        chunk_overlap,
    )
    chunk_embeddings: Optional[np.ndarray] = None
    chunk_norms: Optional[np.ndarray] = None
    if document_chunks and rag_engine.embedding_model is not None:
        chunk_embeddings = rag_engine.embedding_model.encode(
            document_chunks,
            convert_to_numpy=True,
        )
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)

    artifacts: Dict[str, str] = {}
    if case_artifact_dir:
        os.makedirs(case_artifact_dir, exist_ok=True)
        predictions_path = os.path.join(case_artifact_dir, "backend_predictions.json")
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        artifacts["backend_predictions"] = predictions_path

    gt_scores: List[float] = [] # Ground truth numeric scores
    pred_scores: List[float] = [] # Predicted numeric scores
    note_similarities: List[float] = [] # Cosine similarity between GT and predicted notes
    groundedness_records: List[Dict[str, Any]] = []

    for pred in predictions: # Each predicted requirement report
        mapped_id = pred.get("Mapped_ID")
        if not mapped_id or mapped_id not in ground_truth:
            print(f"⚠ Skipping prediction with missing or unmatched Mapped_ID: {mapped_id}")
            continue
        gt_row = ground_truth[mapped_id] # Corresponding GT row for this Mapped_ID
        try:
            if ( gt_row.get("Score") == 'N/A' ):
                print(f"⚠ Skipping GT entry with N/A score for Mapped_ID {mapped_id}.")
                continue
            else:
                gt_score = float(gt_row.get("Score", "0")) # GT score from CSV, default to 0 if missing or invalid
        except ValueError: # (N/A) case is included here
            print(f"⚠ Invalid GT score for Mapped_ID {mapped_id}: {gt_row.get('Score')}.")
            continue

        try:
            if ( pred.get("Score") == 'N/A' ):
                print(f"⚠ Skipping prediction with N/A score for Mapped_ID {mapped_id}.")
                continue
            else:
                pred_score = float(pred.get("Score", "0")) # Predicted score, default to 0 if missing or invalid
        except (TypeError, ValueError):
            print(f"⚠ Invalid predicted score for Mapped_ID {mapped_id}: {pred.get('Score')}.")
            continue

        gt_scores.append(gt_score)
        pred_scores.append(pred_score)

        gt_note = extract_ground_truth_note(gt_row)
        pred_note = pred.get("Auditor_Notes") or pred.get("auditor_notes")
        if gt_note and pred_note:
            try:
                similarity = compute_note_similarity(gt_note, pred_note)
                note_similarities.append(similarity)
            except Exception as exc:
                print(f"⚠ Failed to compute note similarity for {mapped_id}: {exc}")

        contexts = select_relevant_contexts(
            reference_text=pred_note or "",
            chunk_texts=document_chunks,
            chunk_embeddings=chunk_embeddings,
            chunk_norms=chunk_norms,
            top_k=groundedness_top_k,
        )
        question_text = build_requirement_question(mapped_id, pred.get("Requirement_Name"))
        groundedness_records.append(
            {
                "question": question_text,
                "answer": pred_note or "",
                "contexts": contexts,
                "ground_truth": gt_note or "",
                "requirement_id": mapped_id,
                "case": case_name,
            }
        )

    mae = compute_mae(gt_scores, pred_scores) # Compute MAE for this particular case

    mean_note_similarity = (
        sum(note_similarities) / len(note_similarities)
        if note_similarities
        else 0.0
    )

    return (
        {
            "num_pairs": len(gt_scores),
            "mae_score": mae,
            "artifacts": artifacts,
            "note_similarity_count": len(note_similarities),
            "mean_note_similarity": mean_note_similarity,
        },
        groundedness_records,
    )


def main() -> None:
    params = load_params()
    eval_params = params.get("evaluation", {})
    precompute_params = params.get("precompute", {})
    ingestion_params = params.get("ingestion", {})
    groundedness_params = eval_params.get("groundedness", {})

    random_seed = int(eval_params.get("random_seed", 42))

    llm_model = eval_params.get("llm_model")
    llm_temperature = float(eval_params.get("llm_temperature"))
    metrics_output = eval_params.get("metrics_output", "metrics/rag_eval.json")
    gt_cases = eval_params.get("ground_truth", [])
    case_selector = normalize_case_selector(eval_params.get("case_selector"))

    groundedness_top_k = int(groundedness_params.get("top_k", 3))
    groundedness_threshold = float(groundedness_params.get("threshold", 0.75))
    groundedness_chunk_size = int(
        groundedness_params.get(
            "chunk_size",
            ingestion_params.get("chunk_size", 800),
        )
    )
    groundedness_chunk_overlap = int(
        groundedness_params.get(
            "chunk_overlap",
            ingestion_params.get("chunk_overlap", 100),
        )
    )

    # Set seed for reproducibility where possible
    random.seed(random_seed)

    # Prepare metrics dir
    metrics_dir = os.path.dirname(metrics_output)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    # Aggregate metrics across all cases
    all_results: List[Dict[str, Any]] = []
    groundedness_samples: List[Dict[str, Any]] = []

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
    
    if rag_engine is None:
        raise RuntimeError("backend.rag_engine is not available. Run evaluate_rag from the repository root.")

    rag_engine.init_rag()
    print("✓ Using local RAG engine (backend.rag_engine)")

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
            mlflow.log_param("evaluation_mode", "local_engine")
            mlflow.log_param("random_seed", random_seed)
            mlflow.log_param("llm_model", llm_model)
            mlflow.log_param("llm_temperature", llm_temperature)
            mlflow.log_param("precompute_top_k", precompute_params.get("top_k", 3))
            mlflow.log_param("chunk_size", ingestion_params.get("chunk_size"))
            mlflow.log_param("chunk_overlap", ingestion_params.get("chunk_overlap"))
        

        filtered_cases = select_cases(gt_cases, case_selector)

        for case in filtered_cases: # Loop through evaluation cases (can limit with max_eval_cases)
            name = case.get("name", "unknown") # Evaluation case name for logging
            doc_path = case["document_path"] # Path to the document for this case
            report_path = case["report_path"] # Path to the ground truth report for this case
            case_slug = slugify_case_name(name)

            print(f"Evaluating case: {name}")
            if not os.path.exists(doc_path):
                print(f"⚠ Document not found: {doc_path}")
                continue # Skip this case if document is missing
            if not os.path.exists(report_path):
                print(f"⚠ Ground truth report not found: {report_path}")
                continue # Skip this case if ground truth report is missing

            if run_ctx is not None:
                log_case_input_artifacts(name, doc_path, report_path)

            # Use a temporary directory for any artifacts related to this case (like backend predictions)
            with tempfile.TemporaryDirectory(prefix=f"case_{case_slug}_") as tmpdir:
                res, case_groundedness = evaluate_single_case(
                    case_name=name,
                    gt_path=report_path,
                    doc_path=doc_path,
                    chunk_size=groundedness_chunk_size,
                    chunk_overlap=groundedness_chunk_overlap,
                    groundedness_top_k=groundedness_top_k,
                    case_artifact_dir=tmpdir,
                ) # Evaluate this evaluation case
                res["name"] = name # Add case name to results
                all_results.append(res) # Append to all results
                groundedness_samples.extend(case_groundedness)

                if run_ctx is not None:
                    # Log per-case metrics under a prefix
                    mlflow.log_metric(f"{name}_mae_score", res["mae_score"])
                    mlflow.log_metric(f"{name}_num_pairs", res["num_pairs"])
                    mlflow.log_metric(
                        f"{name}_note_similarity",
                        res.get("mean_note_similarity", 0.0),
                    )
                    # Log any artifacts generated during evaluation 
                    artifact_paths = res.get("artifacts", {})
                    for label, file_path in artifact_paths.items():
                        if file_path and os.path.exists(file_path):
                            mlflow.log_artifact(
                                file_path,
                                artifact_path=f"cases/{case_slug}/outputs/{label}",
                            )

        # Compute global aggregates
        if all_results:
            total_pairs = sum(r["num_pairs"] for r in all_results)
            total_note_pairs = sum(r.get("note_similarity_count", 0) for r in all_results)
            # Weighted MAE by number of pairs
            weighted_mae = (
                sum(r["mae_score"] * r["num_pairs"] for r in all_results) / total_pairs
                if total_pairs > 0
                else 0.0
            )
            weighted_note_similarity = (
                sum(
                    r.get("mean_note_similarity", 0.0) * r.get("note_similarity_count", 0)
                    for r in all_results
                ) / total_note_pairs
                if total_note_pairs > 0
                else 0.0
            )
        else:
            total_pairs = 0
            total_note_pairs = 0
            weighted_mae = 0.0
            weighted_note_similarity = 0.0

        summary = {
            "total_cases": len(all_results),
            "total_score_pairs": total_pairs,
            "weighted_mae_score": weighted_mae,
            
            "total_note_pairs": total_note_pairs,
            "mean_note_similarity": weighted_note_similarity,
            
            "cases": all_results,
        }

        groundedness_summary: Optional[Dict[str, Any]] = None
        if groundedness_samples:
            if not RAGAS_AVAILABLE:
                print("⚠ Ragas dependencies not available. Skipping groundedness metric.")
            else:
                try:
                    assert Dataset is not None
                    assert ragas_evaluate is not None
                    assert Groundedness is not None

                    ragas_dataset = Dataset.from_list(
                        [
                            {
                                "question": sample["question"],
                                "answer": sample["answer"],
                                "contexts": sample["contexts"],
                                "ground_truth": sample.get("ground_truth", ""),
                            }
                            for sample in groundedness_samples
                        ]
                    )
                    ragas_result = ragas_evaluate(
                        dataset=ragas_dataset,
                        metrics=[Groundedness()],
                        llm=rag_engine.llm,
                    )
                    groundedness_score = float(ragas_result["groundedness"])
                    groundedness_summary = {
                        "mean_score": groundedness_score,
                        "sample_count": len(groundedness_samples),
                        "threshold": groundedness_threshold,
                    }
                except Exception as exc:
                    print(f"⚠ Failed to compute groundedness metric with Ragas: {exc}")

        if groundedness_summary:
            summary["groundedness"] = groundedness_summary

        # Save metrics to JSON (for DVC)
        with open(metrics_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if run_ctx is not None:
            mlflow.log_metric("score_pairs", total_pairs)
            mlflow.log_metric("weighted_mae_score", weighted_mae)
            
            mlflow.log_metric("note_pairs", total_note_pairs)
            mlflow.log_metric("mean_note_similarity", weighted_note_similarity)
            if groundedness_summary:
                mlflow.log_metric("groundedness_score", groundedness_summary["mean_score"])
                mlflow.log_metric("groundedness_samples", groundedness_summary["sample_count"])
            
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
