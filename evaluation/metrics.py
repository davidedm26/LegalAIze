"""Metrics computation for RAG evaluation."""

from typing import List, Dict, Any, Optional
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
try:
    from datasets import Dataset
except ImportError:
    Dataset = None



try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
except ImportError:
    ragas_evaluate = None
    Faithfulness = None
    AnswerRelevancy = None
    AnswerCorrectness = None



RAGAS_FAITHFULNESS_AVAILABLE = (
    Dataset is not None
    and ragas_evaluate is not None
    and Faithfulness is not None
)
RAGAS_RELEVANCY_AVAILABLE = (
    Dataset is not None
    and ragas_evaluate is not None
    and AnswerRelevancy is not None
)
RAGAS_CORRECTNESS_AVAILABLE = (
    Dataset is not None
    and ragas_evaluate is not None
    and AnswerCorrectness is not None
)


def compute_mae(gt_scores: List[float], pred_scores: List[float]) -> float:
    """Compute Mean Absolute Error between ground truth and predicted scores."""
    if not gt_scores:
        return 0.0
    diffs = [abs(g - p) for g, p in zip(gt_scores, pred_scores)]
    return sum(diffs) / len(diffs)








def compute_ragas_metrics(samples: List[Dict[str, Any]], embedding_model=None) -> Dict[str, Optional[float]]:
    """
    Compute faithfulness scores for a list of samples using a single Ragas evaluation call.
    Args:
        samples: List of evaluation samples
        embedding_model: Optional SentenceTransformer model instance to reuse (avoids reloading)
    Returns a dict with key 'faithfulness'.
    """
    if not samples:
        return {"faithfulness": None}
    if not RAGAS_FAITHFULNESS_AVAILABLE:
        print("⚠ RAGAS faithfulness metric unavailable (check ragas/datasets installation).")
        return {"faithfulness": None}

    import yaml
    import os
    try:
        # Load LLM model name from params.yaml
        params_path = os.path.join(os.path.dirname(__file__), "..", "params.yaml")
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        # Extract the LLM model name from the params
        llm_model = params.get("evaluation", {}).get("llm_model", None)
        llm_temperature = params.get("evaluation", {}).get("llm_temperature", 0.0)
        embedding_model_name = params.get("vectorization", {}).get("model_name", "all-MiniLM-L6-v2")
        
        if llm_model is None:
            print("⚠ LLM model name not found in params.yaml under evaluation.llm_model.")
            return {"faithfulness": None}
        
        # Configure LLM for RAGAS
        llm = ChatOpenAI(model=llm_model, temperature=llm_temperature, request_timeout=180)
        
        # Initialize only Faithfulness metric for sub-requirements
        # Answer Relevancy is problematic in legal context where saying "no evidence found" is valid
        faithfulness_metric = Faithfulness(llm=llm)

        ragas_dataset = Dataset.from_list([
            {
                "question": sample['question'],
                "answer": sample["answer"],
                "contexts": sample["contexts"],
                "ground_truth": sample.get("ground_truth", ""),
            }
            for sample in samples
        ])
        
        print(f"\n🔍 Computing Faithfulness on {len(samples)} sub-requirements...")
        ragas_result = ragas_evaluate(
            dataset=ragas_dataset,
            metrics=[faithfulness_metric],  # Only faithfulness for sub-requirements
            llm=llm,
        )
        df = ragas_result.to_pandas()
        
        # Extract faithfulness score
        if "faithfulness" in df.columns:
            faithfulness = float(df["faithfulness"].mean())
            print(f"  Faithfulness mean: {faithfulness:.4f}")
        elif "nv_response_faithfulness" in df.columns:
            faithfulness = float(df["nv_response_faithfulness"].mean())
            print(f"  Faithfulness mean (nv): {faithfulness:.4f}")
        elif "response_faithfulness" in df.columns:
            faithfulness = float(df["response_faithfulness"].mean())
            print(f"  Faithfulness mean (response): {faithfulness:.4f}")
        else:
            faithfulness = None
            print("  ⚠️ No faithfulness column found!")
        
        # Relevancy and Correctness are computed separately on main requirements
        return {
            "faithfulness": faithfulness
        }
    except Exception as e:
        print(f"⚠ Failed to compute RAGAS metrics: {e}")
        return {"faithfulness": None}


def compute_main_requirement_metrics(
    main_requirement_samples: List[Dict[str, Any]], embedding_model=None
) -> Dict[str, Optional[float]]:
    """
    Compute AnswerCorrectness and AnswerRelevancy for main requirements.
    AnswerCorrectness requires ground truth, AnswerRelevancy works without it.
    
    Args:
        main_requirement_samples: List of main requirement samples
        embedding_model: Optional (not used but kept for consistency)
    
    Returns:
        Dict with keys 'correctness' and 'relevancy', values are scores or None
    """
    if not main_requirement_samples:
        print("  No main requirement samples for metric calculation.")
        return {"correctness": None, "relevancy": None}
    
    # Filter samples that have non-empty ground truth
    samples_with_gt = [
        s for s in main_requirement_samples 
        if s.get("ground_truth") and str(s.get("ground_truth")).strip()
    ]
    
    if not RAGAS_CORRECTNESS_AVAILABLE or not RAGAS_RELEVANCY_AVAILABLE:
        print("⚠ Required RAGAS metrics unavailable (check ragas installation).")
        return {"correctness": None, "relevancy": None}
    
    import yaml
    import os
    try:
        # Load LLM configuration
        params_path = os.path.join(os.path.dirname(__file__), "..", "params.yaml")
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        llm_model = params.get("evaluation", {}).get("llm_model", None)
        llm_temperature = params.get("evaluation", {}).get("llm_temperature", 0.0)
        
        if llm_model is None:
            print("⚠ LLM model name not found in params.yaml under evaluation.llm_model.")
            return {"correctness": None, "relevancy": None}
        
        llm = ChatOpenAI(model=llm_model, temperature=llm_temperature, request_timeout=180)
        embeddings = OpenAIEmbeddings()
        
        # Initialize metrics for main requirements
        correctness_metric = AnswerCorrectness(llm=llm)
        relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
        
        # Customize AnswerRelevancy for compliance audit responses
        try:
            relevancy_metric.question_generation.instruction = """Given a question and answer, create one or more statements from the answer that are relevant to answering the question.
For compliance audit responses, statements identifying gaps, deficiencies, or non-compliance ARE RELEVANT if they directly address compliance status.
Phrases like 'lacks X', 'insufficient Y', 'gaps in Z', 'does not provide', 'absence of' are valid audit findings."""
        except AttributeError:
            pass  # Fallback if prompt customization not available
        
        # Compute on all samples (relevancy doesn't need GT)
        all_samples_dataset = Dataset.from_list([
            {
                "question": sample['question'],
                "answer": sample["answer"],
                "contexts": sample["contexts"],
                "ground_truth": sample.get("ground_truth", ""),
            }
            for sample in main_requirement_samples
        ])
        
        print(f"\n🔍 Computing Relevancy on {len(main_requirement_samples)} main requirements...")
        relevancy_result = ragas_evaluate(
            dataset=all_samples_dataset,
            metrics=[relevancy_metric],
            llm=llm,
        )
        relevancy_df = relevancy_result.to_pandas()
        
        # Extract relevancy score with detailed debugging
        if "answer_relevancy" in relevancy_df.columns:
            relevancy_values = relevancy_df["answer_relevancy"].tolist()
            relevancy = float(relevancy_df["answer_relevancy"].mean())
            print(f"  AnswerRelevancy values: {relevancy_values}")
            print(f"  AnswerRelevancy mean: {relevancy:.4f}")
            print(f"  Non-zero count: {sum(1 for v in relevancy_values if v > 0.001)}/{len(relevancy_values)}")
        else:
            print("  ⚠️ No answer_relevancy column found!")
            relevancy = None
        
        # Compute correctness only on samples with ground truth
        correctness = None
        if samples_with_gt:
            gt_dataset = Dataset.from_list([
                {
                    "question": sample['question'],
                    "answer": sample["answer"],
                    "contexts": sample["contexts"],
                    "ground_truth": sample.get("ground_truth", ""),
                }
                for sample in samples_with_gt
            ])
            
            print(f"🔍 Computing Correctness on {len(samples_with_gt)} main requirements with ground truth...")
            correctness_result = ragas_evaluate(
                dataset=gt_dataset,
                metrics=[correctness_metric],
                llm=llm,
            )
            correctness_df = correctness_result.to_pandas()
            
            if "answer_correctness" in correctness_df.columns:
                correctness = float(correctness_df["answer_correctness"].mean())
                print(f"  AnswerCorrectness mean: {correctness:.4f}")
            else:
                print("  ⚠️ No answer_correctness column found!")
        else:
            print(f"  No samples with ground truth (filtered from {len(main_requirement_samples)} samples) - skipping correctness.")
        
        return {"correctness": correctness, "relevancy": relevancy}
            
    except Exception as e:
        print(f"⚠ Failed to compute main requirement metrics: {e}")
        import traceback
        traceback.print_exc()
        return {"correctness": None, "relevancy": None}
