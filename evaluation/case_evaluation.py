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


def evaluate_single_case(
    *,
    case_name: str,
    gt_path: str,
    doc_path: str,
    case_artifact_dir: Optional[str] = None,
    embedding_model=None,
    ground_truth: bool = False,
    requirement_limit: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate one ground-truth/report pair using the local RAG engine."""
    try:
        from backend import rag_engine
    except ImportError:
        raise RuntimeError("backend.rag_engine is not available. Run evaluate_rag from the project root.")
    
    if ground_truth:
        ground_truth = load_ground_truth_csv(gt_path) # Benchmark report
    else:
        ground_truth = {}
    document_text = load_text(doc_path) # Original document text for RAG evaluation

    if rag_engine is None:
        raise RuntimeError("backend.rag_engine is not available. Run evaluate_rag from the project root.")

    audit_response  = rag_engine.audit_document(document_text, requirement_limit=requirement_limit) # Get predictions (list of RequirementReport objects) from the RAG engine for this document


    # Include the prompt in logged artifacts for downstream analysis and MLflow logging.
    predictions = [report.model_dump() for report in audit_response.requirements]

    # Collect sub-requirement prompts for RAGAS metrics
    sub_ragas_records: List[Dict[str, Any]] = []
    for pred in predictions:
        sub_reqs = pred.get("SubRequirements") or []
        for sub in sub_reqs:
            # Reconstruct the "answer" used for RAGAS (rationale + summary)
            combined_answer = f"{sub.get('Rationale', '')}\n\nSummary: {sub.get('Auditor_Notes', '')}"

            # The prompt/question logic needs to be reconstructed or we rely on contexts
            # Since we didn't save the explicit ragas_question in SubRequirementReport,
            # we will re-generate it here based on the available data.
            req_name = pred.get("Requirement_Name", "Unknown")
            sub_name = sub.get("Reference", "")
            source = sub.get("Source", "")

            
            ragas_question = f"What is the detailed rationale and summary of compliance for the '{req_name}' sub-requirement '{sub_name}' ({source})?"


            contexts = sub.get("Contexts", [])

            # Only add if there is at least some context or a non-trivial answer
            if (contexts and any(c.strip() for c in contexts)) or (combined_answer and combined_answer.strip() and "no information" not in combined_answer.lower()):
                sub_ragas_records.append({
                    "question": ragas_question,
                    "answer": combined_answer,
                    "contexts": contexts,
                    "ground_truth": "",  # Not available at sub-requirement level
                    "requirement_id": pred.get("Requirement_ID", "unknown"),
                    "sub_requirement_name": sub_name,
                    "source": source,
                    "case": case_name,
                })


    # Log input artifacts to MLflow (document and ground truth report) for this case, if MLflow is active. The predictions will be logged as a separate artifact (backend_predictions.json) for easier analysis and debugging.
    artifacts: Dict[str, str] = {}
    if case_artifact_dir:
        os.makedirs(case_artifact_dir, exist_ok=True)
        predictions_path = os.path.join(case_artifact_dir, "backend_predictions.json")
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        artifacts["backend_predictions"] = predictions_path

        # Save aggregation prompt and sub-requirement prompts for each requirement
        for pred in predictions:
            req_id = pred.get("Requirement_ID", "unknown")
            agg_prompt = pred.get("Prompt", "")
            prompt_file = os.path.join(case_artifact_dir, f"prompt_{req_id}.txt")
            with open(prompt_file, "w", encoding="utf-8") as pf:
                pf.write(agg_prompt)
            artifacts[f"prompt_{req_id}"] = prompt_file
            
            # Save sub-requirement prompts
            sub_reqs = pred.get("SubRequirements", [])
            for idx, sub in enumerate(sub_reqs):
                sub_ref = sub.get("Reference", f"sub_{idx}")
                sub_source = sub.get("Source", "unknown")
                sub_prompt = sub.get("Prompt", "")
                if sub_prompt:
                    # Create a safe filename from reference and source
                    safe_ref = sub_ref.replace("/", "_").replace(":", "_").replace(" ", "_")
                    sub_prompt_file = os.path.join(case_artifact_dir, f"prompt_{req_id}_sub_{safe_ref}_{sub_source}.txt")
                    with open(sub_prompt_file, "w", encoding="utf-8") as spf:
                        spf.write(sub_prompt)
                    artifacts[f"prompt_{req_id}_sub_{safe_ref}_{sub_source}"] = sub_prompt_file


    # Initialize accumulators for metrics
    gt_scores: List[float] = []
    pred_scores: List[float] = []
    note_similarities: List[float] = []
    ragas_records: List[Dict[str, Any]] = []

    if  ground_truth:
        # Process each prediction and corresponding ground truth entry, matching by Requirement_ID. If Requirement_ID is missing or does not match any GT entry, skip that prediction and log a warning.
        for pred in predictions:
            requirement_id = pred.get("Requirement_ID")
            if not requirement_id or requirement_id not in ground_truth:
                print(f"⚠ Skipping prediction with missing or unmatched Requirement_ID: {requirement_id}")
                continue

            document_context = pred.get("Context") or []  # Get the context from the prediction, which should be the same as the one used for RAG evaluation. Used for groundedness evaluation in RAGAS, to ensure consistency between the information available at inference time and at evaluation time.

            gt_row = ground_truth[requirement_id]

            gt_score: Optional[float] = None
            pred_score: Optional[float] = None

            # If Score is 'N/A' or missing, we treat it as None and exclude from MAE calculation, but still include in note similarity and groundedness if notes are available. Log warnings for invalid score formats.
            try:
                if gt_row.get("Score") != 'N/A':
                    gt_score = float(gt_row.get("Score", "0"))
            except ValueError:
                print(f"⚠ Invalid GT score for Requirement_ID {requirement_id}: {gt_row.get('Score') }.")

            try:
                if pred.get("Score") != 'N/A':
                    pred_score = float(pred.get("Score", "0"))
            except (TypeError, ValueError):
                print(f"⚠ Invalid predicted score for Requirement_ID {requirement_id}: {pred.get('Score') }.")

            if gt_score is not None and pred_score is not None:
                gt_scores.append(gt_score)
                pred_scores.append(pred_score)

            # Compute note similarity for this couple

            # Extract GT note
            def extract_ground_truth_note(row: Dict[str, Any]) -> Optional[str]:
                """Extract ground truth auditor notes from a CSV row."""
                return (
                    row.get("Auditor Notes")
                    or row.get("auditor_notes")
                    or row.get("Auditor_Notes")
                )
            
            gt_note = extract_ground_truth_note(gt_row)

            auditor_notes = pred.get("Auditor_Notes") or pred.get("auditor_notes")
            #rationale = pred.get("Rationale") or pred.get("rationale")
            #pred_note = auditor_notes + ("\nRationale: " + rationale if rationale else "")
            pred_note = auditor_notes
            if gt_note and pred_note:
                try:
                    similarity = compute_note_similarity(gt_note, pred_note, embedding_model)
                    note_similarities.append(similarity)
                except Exception as exc:
                    print(f"⚠ Failed to compute note similarity for {requirement_id}: {exc}")

            # Build question text for RAGAS groundedness evaluation
            identifier = requirement_id or "Unknown requirement"
            requirement_name = pred.get("Requirement_Name") or gt_row.get("Requirement_Name")
            title = requirement_name 
            # Only use title and id, no metadata
            question_text = f"Is the provided document compliant with the requirement '{title}', according with the provided regulatory chunks from UE AI ACT and ISO standard 42001:2023?"

            # Context is made up by the whole chunks extracted from the Document under Test for the particular requirement. This is the same context that the RAG engine used to produce the prediction, so it allows us to evaluate groundedness in a way that is consistent with the actual information available to the model at inference time.
            # We can reuse the already built context, passing it as a parameter to this function.

            ragas_records.append(
                {
                    "question": question_text,
                    "answer": pred_note or "",
                    # RAGAS expects a list of strings for 'contexts', even if only one context is used
                    "contexts": document_context if isinstance(document_context, list) else [document_context],
                    "ground_truth": gt_note or "",
                    "requirement_id": requirement_id,
                    "case": case_name,
                }
            )
    else:
        # No Ground Truth avaiable
        for pred in predictions:
            document_context = pred.get("Context") or []
            requirement_id = pred.get("Requirement_ID") or "Unknown requirement"
            requirement_name = pred.get("Requirement_Name")
            title = requirement_name or ""
            question_text = f"Is the provided document compliant with the requirement '{title}', according with the provided regulatory chunks from UE AI ACT and ISO standard 42001:2023?"
            
            auditor_notes = pred.get("Auditor_Notes") or pred.get("auditor_notes")
            #rationale = pred.get("Rationale") or pred.get("rationale")
            #pred_note = auditor_notes + ("\nRationale: " + rationale if rationale else "")
            pred_note = auditor_notes
            
            ragas_records.append(
                {
                    "question": question_text,
                    "answer": pred_note or "",
                    "contexts": document_context if isinstance(document_context, list) else [document_context],
                    "ground_truth": "",
                    "requirement_id": requirement_id,
                    "case": case_name,
                }
            )

    # Compute Metrics 
    if ground_truth:
        mae = compute_mae(gt_scores, pred_scores)
        mean_note_similarity = (
            sum(note_similarities) / len(note_similarities)
            if note_similarities
            else 0.0
        )
        ragas_metrics = compute_ragas_metrics(sub_ragas_records)
        case_groundedness_score = ragas_metrics.get("groundedness")
        case_faithfulness_score = ragas_metrics.get("faithfulness")
        case_relevancy_score = ragas_metrics.get("relevancy")
        case_correctness_score = ragas_metrics.get("correctness")

        if case_groundedness_score is None:
            print("⚠ Groundedness score is None, RAGAS evaluation may have failed or is unavailable.")
        if case_faithfulness_score is None:
            print("⚠ Faithfulness score is None, RAGAS evaluation may have failed or is unavailable.")
        if case_relevancy_score is None:
            print("⚠ Relevancy score is None, RAGAS evaluation may have failed or is unavailable.")
        if case_correctness_score is None:
            print("⚠ Correctness score is None, RAGAS evaluation may have failed or is unavailable.")

        return (
            {
                "num_pairs": len(gt_scores),
                "mae_score": mae,
                "artifacts": artifacts,
                "note_similarity_count": len(note_similarities),
                "mean_note_similarity": mean_note_similarity,
                "groundedness_score": case_groundedness_score,
                "groundedness_sample_count": len(sub_ragas_records),
                "faithfulness_score": case_faithfulness_score,
                "faithfulness_sample_count": len(sub_ragas_records),
                "relevancy_score": case_relevancy_score,
                "relevancy_sample_count": len(sub_ragas_records),
                "correctness_score": case_correctness_score,
                "correctness_sample_count": len(sub_ragas_records),
            },
            sub_ragas_records,
        )
    else:
        # No Ground Truth available, we can only compute RAGAS metrics that do not require GT
        ragas_metrics = compute_ragas_metrics(sub_ragas_records)
        case_groundedness_score = ragas_metrics.get("groundedness")
        case_faithfulness_score = ragas_metrics.get("faithfulness")
        case_relevancy_score = ragas_metrics.get("relevancy")

        return (
            {
                "num_pairs": 0,
                "mae_score": None,
                "artifacts": artifacts,
                "note_similarity_count": 0,
                "mean_note_similarity": None,
                "groundedness_score": case_groundedness_score,
                "groundedness_sample_count": len(sub_ragas_records),
                "faithfulness_score": case_faithfulness_score,
                "faithfulness_sample_count": len(sub_ragas_records),
                "relevancy_score": case_relevancy_score,
                "relevancy_sample_count": len(sub_ragas_records),
                "correctness_score": None,
                "correctness_sample_count": 0,
            },
            sub_ragas_records,
        )
