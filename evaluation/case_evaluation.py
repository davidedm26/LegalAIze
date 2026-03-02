"""Case evaluation logic.
Evaluate a single legal compliance case using RAG engine and optional ground truth comparison.
This function performs comprehensive compliance evaluation by:
1. Loading the document and optional ground truth data
2. Running RAG-based audit through the backend engine
3. Computing metrics including MAE (Mean Absolute Error), RAGAS faithfulness, relevancy, and correctness
4. Generating and saving artifact files (predictions, prompts) for MLflow tracking
5. Processing both main requirements and sub-requirements 
Notes:
    - Missing or invalid scores ('N/A' or non-numeric values) are excluded from MAE calculation
    - Sub-requirement rationale is used for faithfulness evaluation
    - Prompts and predictions are saved for MLflow artifact tracking and downstream analysis
    - Ground truth matching is performed by Requirement_ID; unmatched predictions are skipped with warnings
    - Contexts are normalized to lists for RAGAS compatibility
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple

from evaluation.data_loading import load_ground_truth_csv, load_text
from evaluation.metrics import (
    compute_mae,
    compute_subrequirements_ragas_metrics,
    compute_main_requirement_metrics,
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
    
    # Load ground truth if available, otherwise set to empty dict. The ground truth CSV is expected to have a 'Requirement_ID' column that matches the 'Requirement_ID' in the RAG predictions for proper comparison. If the file is missing or invalid, a warning is logged and the evaluation proceeds without GT comparison.
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
            # Use rationale for faithfulness evaluation
            # The rationale provides detailed, grounded analysis
            rationale_answer = sub.get('Rationale', '')

            # Extract sub-requirement reference and source for question formulation and logging
            sub_name = sub.get("Reference", "")
            source = sub.get("Source", "")

            # Question format that matches analytical response style
            ragas_question = f"What is the compliance status of sub-requirement '{sub_name}' from {source}?"


            contexts = sub.get("Contexts", [])

            # Add only if there is at least some context or a non-trivial answer
            if (contexts and any(c.strip() for c in contexts)) or (rationale_answer and rationale_answer.strip() and "no information" not in rationale_answer.lower()):
                sub_ragas_records.append({
                    "question": ragas_question,
                    "answer": rationale_answer,
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


    # Helper function to extract ground truth notes
    def extract_ground_truth_note(row: Dict[str, Any]) -> Optional[str]:
        """Extract ground truth auditor notes from a CSV row."""
        return (
            row.get("Auditor Notes")
            or row.get("auditor_notes")
            or row.get("Auditor_Notes")
        )

    # Initialize accumulators for metrics
    gt_scores: List[float] = []
    pred_scores: List[float] = []
    ragas_records: List[Dict[str, Any]] = []

    # Process predictions - unified loop handles both GT and no-GT cases
    for pred in predictions:
        requirement_id = pred.get("Requirement_ID") or "Unknown requirement"
        document_context = pred.get("Context") or []
        pred_note = pred.get("Auditor_Notes") or pred.get("auditor_notes")
        requirement_name = pred.get("Requirement_Name")
        
        gt_note = ""
        gt_row = None
        
        # If ground truth is available, try to match by Requirement_ID
        if ground_truth:
            if not requirement_id or requirement_id not in ground_truth:
                print(f"⚠ Skipping prediction with missing or unmatched Requirement_ID: {requirement_id}")
                continue
            
            gt_row = ground_truth[requirement_id]
            gt_note = extract_ground_truth_note(gt_row) or ""
            
            # Extract scores for MAE calculation
            gt_score: Optional[float] = None
            pred_score: Optional[float] = None
            
            try:
                if gt_row.get("Score") != 'N/A':
                    gt_score = float(gt_row.get("Score", "0"))
            except ValueError:
                print(f"⚠ Invalid GT score for Requirement_ID {requirement_id}: {gt_row.get('Score')}.")
            
            try:
                if pred.get("Score") != 'N/A':
                    pred_score = float(pred.get("Score", "0"))
            except (TypeError, ValueError):
                print(f"⚠ Invalid predicted score for Requirement_ID {requirement_id}: {pred.get('Score')}.")
            
            if gt_score is not None and pred_score is not None:
                gt_scores.append(gt_score)
                pred_scores.append(pred_score)
            
            # Use GT requirement name if prediction doesn't have it
            if not requirement_name:
                requirement_name = gt_row.get("Requirement_Name")
        
        # Build RAGAS record (same for both GT and no-GT cases)
        title = requirement_name or ""
        question_text = f"Is the provided document compliant with the requirement '{title}', according with the provided regulatory chunks from UE AI ACT and ISO standard 42001:2023?"
        
        # Context is not need for answer relevancy and correctness evaluation.
        ragas_records.append({
            "question": question_text,
            "answer": pred_note or "",
            #"contexts": document_context if isinstance(document_context, list) else [document_context],
            "ground_truth": gt_note,
            "requirement_id": requirement_id,
            "case": case_name,
        })

    # Compute metrics (faithfulness always, relevancy and correctness only with GT)
    sub_metrics = compute_subrequirements_ragas_metrics(sub_ragas_records)
    case_faithfulness_score = sub_metrics.get("faithfulness")
    
    main_metrics = compute_main_requirement_metrics(ragas_records, embedding_model=embedding_model)
    case_relevancy_score = main_metrics.get("relevancy")
    case_correctness_score = main_metrics.get("correctness") if ground_truth else None
    
    # MAE only computed if ground truth available
    mae = compute_mae(gt_scores, pred_scores) if ground_truth else None
    
    # Check for critical failures
    if case_faithfulness_score is None:
        print("⚠ Faithfulness score is None, RAGAS evaluation may have failed.")
    if case_relevancy_score is None:
        print("⚠ AnswerRelevancy is None - RAGAS evaluation may have failed.")
    if ground_truth and case_correctness_score is None:
        print("⚠ AnswerCorrectness is None - check if main requirements have ground truth auditor notes.")

    # Build and return results
    results = {
        "num_pairs": len(gt_scores) if ground_truth else 0,
        "mae_score": mae,
        "artifacts": artifacts,
        "faithfulness_score": case_faithfulness_score,
        "faithfulness_sample_count": len(sub_ragas_records),
        "relevancy_score": case_relevancy_score,
        "relevancy_sample_count": len(ragas_records),
        "correctness_score": case_correctness_score,
        "correctness_sample_count": len(ragas_records) if ground_truth else 0,
    }
    
    return results, sub_ragas_records, ragas_records
