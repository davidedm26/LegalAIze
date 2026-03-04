"""
RAG Evaluation Script
This script evaluates the RAG system using the local RAG engine defined in backend.rag_engine. It loads evaluation cases defined in params.yaml, runs them through the RAG engine, computes metrics and logs results to MLflow (if configured).    
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sentence_transformers import SentenceTransformer
import os
import json
import random
import tempfile
from typing import Any, Dict, List
import requests
from dotenv import load_dotenv
import mlflow

from evaluation import (
    load_params,
    evaluate_single_case,
    normalize_case_selector,
    select_cases,
    slugify_case_name,
    log_case_input_artifacts,
)

try:
    from backend import rag_engine
except ImportError:
    rag_engine = None

load_dotenv()

# Configure MLflow if tracking URI is set via environment variable, otherwise check for DagsHub credentials to construct a tracking URI. If neither is available, MLflow logging will be disabled and a warning will be printed.
def setup_mlflow():
    # 1. Primary Attempt: Use MLFLOW_TRACKING_URI from environment variable ( Remember to turn on MLFlow )
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    
    # If the URI is local, check if the server responds
    if mlflow_uri:
        try:
            mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
            mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")     

            requests.get(mlflow_uri.rstrip('/') + '/health', timeout=2)

            if mlflow_username and mlflow_password:
                mlflow.set_tracking_uri(mlflow_uri)
                mlflow.set_tracking_username(mlflow_username)
                mlflow.set_tracking_password(mlflow_password)

            print(f"✅ MLflow connected to: {mlflow_uri}")
            return
        except:
            print(f"⚠️ Server at URI {mlflow_uri} not reachable.")

    # 2. Attempt: DagsHub (Fallback)
    dagshub_user = os.getenv("DAGSHUB_USERNAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    repo_name = "LegalAIze" # Repo name

    if dagshub_user and dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        remote_uri = f"https://dagshub.com/{dagshub_user}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(remote_uri)
        print(f"🌐 MLflow connected to DagsHub: {remote_uri}")
    else:
        print("⚠️ MLFLOW_TRACKING_URI not set and DagsHub credentials not found. MLflow logging will be disabled.")

setup_mlflow()

def main() -> None:

    # Load parameters from params.yaml
    params = load_params()
    eval_params = params.get("evaluation", {})
    precompute_params = params.get("precompute", {})
    vect_params = params.get("vectorization", {})
    ingestion_params = params.get("ingestion", {})

    random_seed = int(eval_params.get("random_seed", 42))
  
    llm_model = eval_params.get("llm_model")
    llm_temperature = float(eval_params.get("llm_temperature"))
    metrics_output = eval_params.get("metrics_output", "metrics/rag_eval.json")
    debug_output_dir = eval_params.get("debug_output_dir", "data/debug")
    gt_cases = eval_params.get("ground_truth", [])
    case_selector = normalize_case_selector(eval_params.get("case_selector"))
    requirement_limit = eval_params.get("requirement_limit", None)

    # Initialize embedding model for evaluation
    embedding_model = SentenceTransformer(vect_params.get("model_name", "all-MiniLM-L6-v2"))

    # Set seed for reproducibility where possible
    random.seed(random_seed)

    # Prepare metrics dir (useful when MLflow is not configured and local JSON output is needed for metrics storage, e.g., for DVC tracking)
    metrics_dir = os.path.dirname(metrics_output)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    
    # Prepare debug output dir for saving audit reports
    if debug_output_dir:
        os.makedirs(debug_output_dir, exist_ok=True)
        print(f"📁 Debug reports will be saved to: {debug_output_dir}")

    # Aggregate metrics across all cases
    all_results: List[Dict[str, Any]] = []
    sub_ragas_records: List[Dict[str, Any]] = []  # Sub-requirements (for faithfulness)
    main_ragas_records: List[Dict[str, Any]] = []  # Main requirements (for relevancy/correctness)

    experiment_name = eval_params.get("mlflow_experiment", "rag_evaluation")

    if mlflow is not None:
        mlflow.set_experiment(experiment_name)

    # Start MLflow run context (if MLflow is configured properly, otherwise this will be None and logging calls will be skipped)
    run_ctx = (
        mlflow.start_run(run_name="rag_eval")
        if mlflow is not None
        else None
    )

    
    if rag_engine is None:
        raise RuntimeError("backend.rag_engine is not available. Run evaluate_rag from the repository root.")

    # Initialize RAG engine components (like retriever, LLM, etc.) before running evaluations. This ensures that the RAG system is ready to process the evaluation cases. The initialization logic is defined in backend.rag_engine.init_rag(), which is also called on startup in the FastAPI app lifespan event.
    rag_engine.init_rag()
    print("✓ Using local RAG engine (backend.rag_engine)")

    mapping_path = ingestion_params.get(
        "mapping_path",
        os.path.join(ingestion_params.get("data_dir", "data"), "mapping.json"),
    )


    if run_ctx is not None:
        print(f"✓ MLflow run started: {run_ctx.info.run_id}")
        # Set run label (run_name) and GitHub tags if running in GitHub Actions environment
        if os.getenv("GITHUB_ACTIONS", "false").lower() == "true":
            mlflow.set_tag("mlflow.runName", "github_actions")
            mlflow.set_tag("github_workflow", os.getenv("GITHUB_WORKFLOW", ""))
            mlflow.set_tag("github_run_id", os.getenv("GITHUB_RUN_ID", ""))
            mlflow.set_tag("github_run_number", os.getenv("GITHUB_RUN_NUMBER", ""))
            mlflow.set_tag("github_job", os.getenv("GITHUB_JOB", ""))
            mlflow.set_tag("github_ref", os.getenv("GITHUB_REF", ""))
            mlflow.set_tag("github_sha", os.getenv("GITHUB_SHA", ""))
            mlflow.set_tag("github_actor", os.getenv("GITHUB_ACTOR", ""))
            mlflow.set_tag("github_repository", os.getenv("GITHUB_REPOSITORY", ""))
        else:
            mlflow.set_tag("mlflow.runName", "local")
    else:
        print("⚠ MLflow tracking URI not configured. Metrics will not be logged to MLflow.")

    try:
        # Log static params to MLflow if available
        if run_ctx is not None:
            mlflow.log_param("evaluation_mode", "local_engine")
            mlflow.log_param("random_seed", random_seed)
            mlflow.log_param("llm_model", llm_model)
            mlflow.log_param("llm_temperature", llm_temperature)
            mlflow.log_param("embedding_model",vect_params.get("model_name", ""))
            mlflow.log_param("precompute_top_k", precompute_params.get("top_k", 3))
            mlflow.log_param("requirement_limit", requirement_limit if requirement_limit is not None else "all")
            # Log the actual chunk_size and chunk_overlap used by the backend
            mlflow.log_param("chunk_size", rag_engine.rag_params.get("document_chunk_size"))
            mlflow.log_param("chunk_overlap", rag_engine.rag_params.get("document_chunk_overlap"))

            # Log prompt templates to MLflow after run initialization
            try:
                # Use the EvaluationEngine from rag_engine to access prompt templates
                if rag_engine.evaluation_engine:
                    # Example values for template logging
                    example_reference = "EXAMPLE_REFERENCE"
                    example_content = "EXAMPLE_CONTENT"
                    example_chunks = ["EXAMPLE_CHUNK_1", "EXAMPLE_CHUNK_2"]

                    # Access methods to log prompt templates
                    sub_prompt_template = rag_engine.evaluation_engine._get_sub_prompt("EXAMPLE_MAIN_REQ", example_reference, "EXAMPLE_SOURCE", example_content, example_chunks)
                    sub_prompt_file = os.path.join(tempfile.gettempdir(), "sub_prompt_template.txt")
                    with open(sub_prompt_file, "w", encoding="utf-8") as f:
                        f.write(sub_prompt_template)
                    mlflow.log_artifact(sub_prompt_file, artifact_path="prompt_templates")

                    agg_prompt_template = rag_engine.evaluation_engine._get_aggregate_prompt([
                        {"reference": example_reference, "score": 5, "answer": "Good"}
                    ], computed_score=3.5)
                    agg_prompt_file = os.path.join(tempfile.gettempdir(), "aggregate_prompt_template.txt")
                    with open(agg_prompt_file, "w", encoding="utf-8") as f:
                        f.write(agg_prompt_template)
                    mlflow.log_artifact(agg_prompt_file, artifact_path="prompt_templates")
            except Exception as e:
                print(f"⚠ Failed to log prompt templates to MLflow: {e}")

        # Select evaluation cases based on case_selector
        filtered_cases = select_cases(gt_cases, case_selector) # Select cases based on case_selector (params)

        
        # Loop through evaluation cases
        for i, case in enumerate(filtered_cases):
            ground_truth_present = True

            name = case.get("name", "unknown") # Evaluation case name for logging
            doc_path = case["document_path"] # Path to the document for this case
            report_path = case.get("report_path", None) # Path to the ground truth report for this case
            case_slug = slugify_case_name(name) # Create a slug for this case to use in artifact paths and logging

            print(f"Evaluating case: {name}")
            if not os.path.exists(doc_path):
                print(f"⚠ Document not found: {doc_path}")
                continue
            if report_path is None or not os.path.exists(report_path):
                print(f"⚠ Ground truth report not found, Score MAE, Note Similarity and RAGAS Correctness metrics will be skipped for case: {doc_path}")
                report_path_for_eval = None
                ground_truth_present = False
            else:
                report_path_for_eval = report_path

            if run_ctx is not None:
                log_case_input_artifacts(name, doc_path, report_path) # Log input documents (documentation and ground truth report) to MLflow for this case

            # Use a temporary directory for any artifacts related to this case (like backend predictions)
            with tempfile.TemporaryDirectory(prefix=f"case_{case_slug}_") as tmpdir:
                res, case_sub_ragas_records, case_main_ragas_records = evaluate_single_case(
                    case_name=name,
                    gt_path=report_path_for_eval,
                    doc_path=doc_path,
                    case_artifact_dir=tmpdir,
                    embedding_model=embedding_model,
                    ground_truth=ground_truth_present,
                    requirement_limit=requirement_limit,
                ) # Evaluate this evaluation case
                res["name"] = name # Add case name to results
                all_results.append(res) # Append to all results
                
                # Save audit report to debug directory
                if debug_output_dir and "artifacts" in res:
                    backend_pred_path = res["artifacts"].get("backend_predictions")
                    if backend_pred_path and os.path.exists(backend_pred_path):
                        debug_report_path = os.path.join(debug_output_dir, f"audit_{case_slug}.json")
                        with open(backend_pred_path, "r", encoding="utf-8") as src:
                            predictions_data = json.load(src)
                        with open(debug_report_path, "w", encoding="utf-8") as dst:
                            json.dump(predictions_data, dst, indent=2, ensure_ascii=False)
                        print(f"  💾 Saved audit report to: {debug_report_path}")

                # Collect RAGAS records for logging
                sub_ragas_records.extend(case_sub_ragas_records)  # Sub-requirements
                main_ragas_records.extend(case_main_ragas_records)  # Main requirements

                if run_ctx is not None:
                    # Log per-case metrics with a step index (for easier aggregation in MLflow UI charts)
                    if res.get("mae_score") is not None:
                        mlflow.log_metric("mae_score_list", res["mae_score"], step=i)
                    if res.get("faithfulness_score") is not None:
                        mlflow.log_metric("faithfulness_list", res["faithfulness_score"], step=i)
                    if res.get("relevancy_score") is not None:
                        mlflow.log_metric("relevancy_list", res["relevancy_score"], step=i)
                    if res.get("correctness_score") is not None:
                        mlflow.log_metric("correctness_list", res["correctness_score"], step=i)


                    # Log any artifacts produced during evaluation (like model predictions, intermediate files, etc.) for this case
                    artifact_paths = res.get("artifacts", {})
                    for label, file_path in artifact_paths.items():
                        if file_path and os.path.exists(file_path):
                            mlflow.log_artifact(
                                file_path,
                                artifact_path=f"cases/{case_slug}/outputs/{label}",
                            )

        # Compute aggregated metrics across all results and log them to MLflow and/or save to a JSON file for DVC tracking.
        if all_results:
            total_score_pairs = sum(r["num_pairs"] for r in all_results)
            total_faithfulness_samples = sum(r.get("faithfulness_sample_count", 0) for r in all_results)
            total_relevancy_samples = sum(r.get("relevancy_sample_count", 0) for r in all_results)
            total_correctness_samples = sum(r.get("correctness_sample_count", 0) for r in all_results)

            # Weighted MAE by number of pairs
            weighted_mae = (
                sum((r["mae_score"] or 0) * r["num_pairs"] for r in all_results if r.get("num_pairs") is not None) / total_score_pairs
                if total_score_pairs > 0
                else 0.0
            )

            # Weighted faithfulness by sample count
            weighted_faithfulness = None
            faithfulness_results = [r for r in all_results if r.get("faithfulness_score") is not None]
            if faithfulness_results and total_faithfulness_samples > 0:
                weighted_faithfulness = (
                    sum((r.get("faithfulness_score") or 0) * (r.get("faithfulness_sample_count") or 0) for r in faithfulness_results) / total_faithfulness_samples
                )

            # Weighted relevancy by sample count
            weighted_relevancy = None
            relevancy_results = [r for r in all_results if r.get("relevancy_score") is not None]
            if relevancy_results and total_relevancy_samples > 0:
                weighted_relevancy = (
                    sum((r.get("relevancy_score") or 0) * (r.get("relevancy_sample_count") or 0) for r in relevancy_results) / total_relevancy_samples
                )

            # Weighted correctness by sample count
            weighted_correctness = None
            correctness_results = [r for r in all_results if r.get("correctness_score") is not None]
            if correctness_results and total_correctness_samples > 0:
                weighted_correctness = (
                    sum((r.get("correctness_score") or 0) * (r.get("correctness_sample_count") or 0) for r in correctness_results) / total_correctness_samples
                )

        else: # Fallback values if no results were processed (e.g., all cases were skipped due to missing files)
            total_score_pairs = 0
            total_faithfulness_samples = 0
            total_relevancy_samples = 0
            total_correctness_samples = 0
            weighted_mae = 0.0
            weighted_faithfulness = None
            weighted_relevancy   = None
            weighted_correctness = None

        summary = {
            "total_cases": len(all_results),
            "total_score_pairs": total_score_pairs,
            "weighted_mae_score": weighted_mae,
            "total_faithfulness_samples": total_faithfulness_samples,
            "total_relevancy_samples": total_relevancy_samples,
            "total_correctness_samples": total_correctness_samples,
            "mean_faithfulness_score": weighted_faithfulness,
            "mean_relevancy_score": weighted_relevancy,
            "mean_correctness_score": weighted_correctness,
            "cases": all_results,
        }

        # Save metrics to JSON (for local evaluation without MLflow)
        with open(metrics_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if run_ctx is not None:
            mlflow.log_metric("mae_score_pairs", total_score_pairs)
            if weighted_mae is not None:
                mlflow.log_metric("mae_weighted_score", weighted_mae)

            if weighted_faithfulness is not None:
                mlflow.log_metric("faithfulness_score", weighted_faithfulness)
                mlflow.log_metric("faithfulness_samples", total_faithfulness_samples)
            if weighted_relevancy is not None:
                mlflow.log_metric("relevancy_score", weighted_relevancy)
                mlflow.log_metric("relevancy_samples", total_relevancy_samples)
            if weighted_correctness is not None:
                mlflow.log_metric("correctness_score", weighted_correctness)
                mlflow.log_metric("correctness_samples", total_correctness_samples)

            # Log the metrics file also as artifact
            mlflow.log_artifact(metrics_output)
            if os.path.exists(mapping_path):
                mlflow.log_artifact(mapping_path, artifact_path="inputs")

            # Log RAGAS records as artifacts for potential further analysis
            # Sub-requirements records (for faithfulness evaluation)
            if sub_ragas_records:
                sub_ragas_path = os.path.join(metrics_dir, "ragas_sub_requirements.json")
                with open(sub_ragas_path, "w", encoding="utf-8") as f:
                    json.dump(sub_ragas_records, f, indent=2, ensure_ascii=False)
                mlflow.log_artifact(sub_ragas_path, artifact_path="ragas")
                print(f"✓ Logged {len(sub_ragas_records)} sub-requirement records to MLflow")
            
            # Main requirements records (for relevancy/correctness evaluation)
            if main_ragas_records:
                main_ragas_path = os.path.join(metrics_dir, "ragas_main_requirements.json")
                with open(main_ragas_path, "w", encoding="utf-8") as f:
                    json.dump(main_ragas_records, f, indent=2, ensure_ascii=False)
                mlflow.log_artifact(main_ragas_path, artifact_path="ragas")
                print(f"✓ Logged {len(main_ragas_records)} main requirement records to MLflow")

        print("✓ RAG evaluation completed.")

    finally:
        # Clean up resources to avoid shutdown warnings
        if rag_engine and hasattr(rag_engine, 'vector_db') and rag_engine.vector_db:
            try:
                rag_engine.vector_db.close()
            except Exception:
                pass  # Ignore errors during cleanup
        
        if run_ctx is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
