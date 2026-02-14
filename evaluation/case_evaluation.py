"""Case evaluation logic."""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from evaluation.data_loading import load_ground_truth_csv, load_text
from evaluation.metrics import (
    compute_mae,
    compute_note_similarity,
    compute_ragas_metrics,
)
from evaluation.preprocessing import (
    split_document_for_groundedness,
    build_requirement_question,
    select_relevant_contexts,
    extract_ground_truth_note,
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
    embedding_model=None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate one ground-truth/report pair using the local RAG engine."""
    try:
        from backend import rag_engine
    except ImportError:
        raise RuntimeError("backend.rag_engine is not available. Run evaluate_rag from the project root.")
    
    ground_truth = load_ground_truth_csv(gt_path) # Benchmark report
    document_text = load_text(doc_path) # Original document text for RAG evaluation

    if rag_engine is None:
        raise RuntimeError("backend.rag_engine is not available. Run evaluate_rag from the project root.")

    audit_response = rag_engine.audit_document(document_text) # Get predictions

    # Exclude the prompt from logged artifacts, it may contain sensitive info and is not needed for evaluation analysis. Only log the structured predictions.
    predictions = [report.model_dump(exclude={"Prompt"}) for report in audit_response.requirements]

    # Split input document into chunks for groundedness evaluation
    document_chunks = split_document_for_groundedness(
        document_text,
        chunk_size,
        chunk_overlap,
    )

    chunk_embeddings: Optional[np.ndarray] = None
    chunk_norms: Optional[np.ndarray] = None
    if document_chunks and embedding_model is not None: # Precompute embeddings for document chunks
        chunk_embeddings = embedding_model.encode(
            document_chunks,
            convert_to_numpy=True,
        )
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1) # Precompute norms for efficient cosine similarity calculation during context selection

    # Log input artifacts to MLflow (document and ground truth report) for this case, if MLflow is active. The predictions will be logged as a separate artifact (backend_predictions.json) for easier analysis and debugging.
    artifacts: Dict[str, str] = {}
    if case_artifact_dir:
        os.makedirs(case_artifact_dir, exist_ok=True)
        predictions_path = os.path.join(case_artifact_dir, "backend_predictions.json")
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        artifacts["backend_predictions"] = predictions_path


    # Initialize accumulators for metrics
    gt_scores: List[float] = []
    pred_scores: List[float] = []
    note_similarities: List[float] = []
    ragas_records: List[Dict[str, Any]] = []

    # Process each prediction and corresponding ground truth entry, matching by Mapped_ID. If Mapped_ID is missing or does not match any GT entry, skip that prediction and log a warning.
    for pred in predictions:
        mapped_id = pred.get("Mapped_ID")
        if not mapped_id or mapped_id not in ground_truth:
            print(f"⚠ Skipping prediction with missing or unmatched Mapped_ID: {mapped_id}")
            continue
        gt_row = ground_truth[mapped_id]

        gt_score: Optional[float] = None
        pred_score: Optional[float] = None

        # If Score is 'N/A' or missing, we treat it as None and exclude from MAE calculation, but still include in note similarity and groundedness if notes are available. Log warnings for invalid score formats.
        try:
            if gt_row.get("Score") != 'N/A':
                gt_score = float(gt_row.get("Score", "0"))
        except ValueError:
            print(f"⚠ Invalid GT score for Mapped_ID {mapped_id}: {gt_row.get('Score')}.")

        try:
            if pred.get("Score") != 'N/A':
                pred_score = float(pred.get("Score", "0"))
        except (TypeError, ValueError):
            print(f"⚠ Invalid predicted score for Mapped_ID {mapped_id}: {pred.get('Score')}.")

        if gt_score is not None and pred_score is not None:
            gt_scores.append(gt_score)
            pred_scores.append(pred_score)

        # Compute note similarity for this couple
        gt_note = extract_ground_truth_note(gt_row)
        pred_note = pred.get("Auditor_Notes") or pred.get("auditor_notes")
        if gt_note and pred_note:
            try:
                similarity = compute_note_similarity(gt_note, pred_note, embedding_model)
                note_similarities.append(similarity)
            except Exception as exc:
                print(f"⚠ Failed to compute note similarity for {mapped_id}: {exc}")


        # To be consisten with the RAG worflow we will use the whole document as context for RAGAS evaluation
        """
        contexts = select_relevant_contexts(
            reference_text=pred_note or "",
            chunk_texts=document_chunks,
            chunk_embeddings=chunk_embeddings,
            chunk_norms=chunk_norms,
            top_k=groundedness_top_k,
            embedding_model=embedding_model,
        )
        """

        # Build question text for RAGAS groundedness evaluation, including requirement metadata if available
        question_text = build_requirement_question(mapped_id, pred.get("Requirement_Name"))

        ragas_records.append(
            {
                "question": question_text,
                "answer": pred_note or "",
                # RAGAS expects a list of strings for 'contexts', even if only one context is used
                "contexts": [document_text],
                "ground_truth": gt_note or "",
                "requirement_id": mapped_id,
                "case": case_name,
            }
        )

    # Compute Metrics 

    mae = compute_mae(gt_scores, pred_scores)

    mean_note_similarity = (
        sum(note_similarities) / len(note_similarities)
        if note_similarities
        else 0.0
    )

    ragas_metrics = compute_ragas_metrics(ragas_records)

    case_groundedness_score = ragas_metrics.get("groundedness")
    case_faithfulness_score = ragas_metrics.get("faithfulness")

    if case_groundedness_score is None:
        print("⚠ Groundedness score is None, RAGAS evaluation may have failed or is unavailable.")
    if case_faithfulness_score is None:
        print("⚠ Faithfulness score is None, RAGAS evaluation may have failed or is unavailable.")


    return (
        {
            "num_pairs": len(gt_scores),
            "mae_score": mae,
            "artifacts": artifacts,
            "note_similarity_count": len(note_similarities),
            "mean_note_similarity": mean_note_similarity,
            "groundedness_score": case_groundedness_score,
            "groundedness_sample_count": len(ragas_records),
            "faithfulness_score": case_faithfulness_score,
            "faithfulness_sample_count": len(ragas_records),
        },
        ragas_records,
    )
