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
    params = load_params()
    eval_params = params.get("evaluation", {})
    precompute_params = params.get("precompute", {})
    vect_params = params.get("vectorization", {})
    ingestion_params = params.get("ingestion", {})
    groundedness_params = eval_params.get("groundedness", {})

    random_seed = int(eval_params.get("random_seed", 42))

    llm_model = eval_params.get("llm_model")
    llm_temperature = float(eval_params.get("llm_temperature"))
    metrics_output = eval_params.get("metrics_output", "metrics/rag_eval.json")
    gt_cases = eval_params.get("ground_truth", [])
    case_selector = normalize_case_selector(eval_params.get("case_selector"))




    embedding_model = SentenceTransformer(vect_params.get("model_name", "all-MiniLM-L6-v2"))

    # Set seed for reproducibility where possible
    random.seed(random_seed)

    # Prepare metrics dir ( useful when MLflow is not configured and we rely on local JSON output for metrics storage, e.g., for DVC tracking )
    metrics_dir = os.path.dirname(metrics_output)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    # Aggregate metrics across all cases
    all_results: List[Dict[str, Any]] = []
    groundedness_samples: List[Dict[str, Any]] = []

    experiment_name = eval_params.get("mlflow_experiment", "rag_evaluation")

    if mlflow is not None:
        mlflow.set_experiment(experiment_name)

    run_ctx = (
        mlflow.start_run(run_name="rag_eval")
        if mlflow is not None
        else None
    )

    prompt_template_path = None

    # Save prompt template locally for next steps (artifact logging) if available in rag_engine.
    if run_ctx is not None and rag_engine is not None:
        with tempfile.NamedTemporaryFile("w", suffix="_prompt_template.txt", delete=False, encoding="utf-8") as f:
            f.write(getattr(rag_engine, "PROMPT_TEMPLATE", "").strip())
            prompt_template_path = f.name
    
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
            mlflow.log_param("embedding_model",vect_params.get("model_name", ""))
            mlflow.log_param("precompute_top_k", precompute_params.get("top_k", 3))
            mlflow.log_param("chunk_size", ingestion_params.get("chunk_size"))
            mlflow.log_param("chunk_overlap", ingestion_params.get("chunk_overlap"))
            if prompt_template_path and os.path.exists(prompt_template_path):
                mlflow.log_artifact(prompt_template_path, artifact_path="inputs")
        
            
        filtered_cases = select_cases(gt_cases, case_selector) # Select cases based on case_selector (params)

        # Loop through evaluation cases
        for i, case in enumerate(filtered_cases):
            name = case.get("name", "unknown") # Evaluation case name for logging
            doc_path = case["document_path"] # Path to the document for this case
            report_path = case["report_path"] # Path to the ground truth report for this case
            case_slug = slugify_case_name(name) # Create a slug for this case to use in artifact paths and logging

            print(f"Evaluating case: {name}")
            if not os.path.exists(doc_path):
                print(f"⚠ Document not found: {doc_path}")
                continue # Skip this case if document is missing
            if not os.path.exists(report_path):
                print(f"⚠ Ground truth report not found: {report_path}")
                continue # Skip this case if ground truth report is missing

            if run_ctx is not None:
                log_case_input_artifacts(name, doc_path, report_path) # Log input documents (documentation and ground truth report) to MLflow for this case

            # Use a temporary directory for any artifacts related to this case (like backend predictions)
            with tempfile.TemporaryDirectory(prefix=f"case_{case_slug}_") as tmpdir:
                res, case_groundedness = evaluate_single_case(
                    case_name=name,
                    gt_path=report_path,
                    doc_path=doc_path,
                    case_artifact_dir=tmpdir,
                    embedding_model=embedding_model,
                ) # Evaluate this evaluation case
                res["name"] = name # Add case name to results
                all_results.append(res) # Append to all results

                groundedness_samples.extend(case_groundedness) # Collect groundedness samples for potential further analysis or separate logging

                if run_ctx is not None:
                    # Log per-case metrics with a step index (for easier aggregation in MLflow UI charts)
                    mlflow.log_metric("mae_score_list", res["mae_score"], step=i)
                    mlflow.log_metric("note_similarity_list",res.get("mean_note_similarity", 0.0),step=i)
                    if res.get("groundedness_score") is not None:
                        mlflow.log_metric("groundedness_list", res["groundedness_score"], step=i)
                    if res.get("faithfulness_score") is not None:
                        mlflow.log_metric("faithfulness_list", res["faithfulness_score"], step=i)

                    # Log any artifacts produced during evaluation (like model predictions, intermediate files, etc.) for this case
                    artifact_paths = res.get("artifacts", {})
                    for label, file_path in artifact_paths.items():
                        if file_path and os.path.exists(file_path):
                            mlflow.log_artifact(
                                file_path,
                                artifact_path=f"eval_cases/{case_slug}/outputs/{label}",
                            )

        # After processing all cases, compute aggregated metrics across all results and log them to MLflow and/or save to a JSON file for DVC tracking or other uses.
        if all_results:
            total_score_pairs = sum(r["num_pairs"] for r in all_results)
            total_note_pairs = sum(r.get("note_similarity_count", 0) for r in all_results)
            total_groundedness_samples = sum(r.get("groundedness_sample_count", 0) for r in all_results)
            total_faithfulness_samples = sum(r.get("faithfulness_sample_count", 0) for r in all_results)
            
            # Weighted MAE by number of pairs
            weighted_mae = (
                sum(r["mae_score"] * r["num_pairs"] for r in all_results) / total_score_pairs
                if total_score_pairs > 0
                else 0.0
            )

            # Weighted note similarity by number of note pairs
            weighted_note_similarity = (
                sum(
                    r.get("mean_note_similarity", 0.0) * r.get("note_similarity_count", 0)
                    for r in all_results
                ) / total_note_pairs
                if total_note_pairs > 0
                else 0.0
            )

            # Weighted groundedness by sample count
            weighted_groundedness = None
            groundedness_results = [r for r in all_results if r.get("groundedness_score") is not None]
            if groundedness_results and total_groundedness_samples > 0:
                weighted_groundedness = (
                    sum(
                        r["groundedness_score"] * r.get("groundedness_sample_count", 0)
                        for r in groundedness_results
                    ) / total_groundedness_samples
                )

            # Weighted faithfulness by sample count
            weighted_faithfulness = None
            faithfulness_results = [r for r in all_results if r.get("faithfulness_score") is not None]
            if faithfulness_results and total_faithfulness_samples > 0:
                weighted_faithfulness = (
                    sum(
                        r["faithfulness_score"] * r.get("faithfulness_sample_count", 0)
                        for r in faithfulness_results
                    ) / total_faithfulness_samples
                )

        else: # Fallback values if no results were processed (e.g., all cases were skipped due to missing files)
            total_score_pairs = 0
            total_note_pairs = 0
            total_groundedness_samples = 0
            total_faithfulness_samples = 0
            weighted_mae = 0.0
            weighted_note_similarity = 0.0
            weighted_groundedness = None
            weighted_faithfulness = None

        summary = {
            "total_cases": len(all_results),
            "total_score_pairs": total_score_pairs,
            "weighted_mae_score": weighted_mae,
            "total_note_pairs": total_note_pairs,
            "mean_note_similarity": weighted_note_similarity,
            "total_groundedness_samples": total_groundedness_samples,
            "total_faithfulness_samples": total_faithfulness_samples,
            "mean_groundedness_score": weighted_groundedness,
            "mean_faithfulness_score": weighted_faithfulness,
            "cases": all_results,
        }

        # Save metrics to JSON (for local evaluation without MLflow)
        with open(metrics_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if run_ctx is not None:
            mlflow.log_metric("mae_score_pairs", total_score_pairs)
            mlflow.log_metric("mae_weighted_score", weighted_mae)
            mlflow.log_metric("note_similarity_pairs", total_note_pairs)
            mlflow.log_metric("note_similarity_mean", weighted_note_similarity)
            
            if weighted_groundedness is not None:
                mlflow.log_metric("groundedness_score", weighted_groundedness)
                mlflow.log_metric("groundedness_samples", total_groundedness_samples)
            if weighted_faithfulness is not None:
                mlflow.log_metric("faithfulness_score", weighted_faithfulness)
                mlflow.log_metric("faithfulness_samples", total_faithfulness_samples)
            
            # Log the metrics file also as artifact
            mlflow.log_artifact(metrics_output)
            if os.path.exists(mapping_path):
                mlflow.log_artifact(mapping_path, artifact_path="inputs")

        print("✓ RAG evaluation completed.")

    finally:
        if run_ctx is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
