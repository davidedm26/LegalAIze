"""Metrics computation for RAG evaluation."""

from typing import List, Dict, Any, Optional
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


# ============================================================================
# RAGAS 0.4.0 has issues parsing JSON wrapped in markdown code fences (```json...```)
# This patch intercepts json.loads calls to remove markdown formatting
import json as json_module

def _clean_json_string(s: str) -> str:
    """Remove markdown code fences from JSON strings."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    # Remove ```json and ``` markers
    if s.startswith("```json"):
        s = s[7:]  # Remove ```json
    elif s.startswith("```"):
        s = s[3:]  # Remove ```
    if s.endswith("```"):
        s = s[:-3]  # Remove trailing ```
    return s.strip()

# Store original json.loads
_original_json_loads = json_module.loads

def _patched_json_loads(s, *args, **kwargs):
    """Patched json.loads that handles markdown code fences."""
    if isinstance(s, str):
        s = _clean_json_string(s)
    return _original_json_loads(s, *args, **kwargs)

# Apply monkey-patch to json module
json_module.loads = _patched_json_loads
print("Applied JSON markdown code fence patch for RAGAS compatibility")
# ============================================================================


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



def compute_subrequirements_ragas_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
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
        
        # Extract faithfulness score (excluding zeros from average)
        if "faithfulness" in df.columns:
            non_zero_values = df["faithfulness"][df["faithfulness"] > 0.001]
            faithfulness = float(non_zero_values.mean()) if len(non_zero_values) > 0 else 0.0
            print(f"  Faithfulness mean: {faithfulness:.4f} (non-zero count: {len(non_zero_values)}/{len(df)})")
        elif "nv_response_faithfulness" in df.columns:
            non_zero_values = df["nv_response_faithfulness"][df["nv_response_faithfulness"] > 0.001]
            faithfulness = float(non_zero_values.mean()) if len(non_zero_values) > 0 else 0.0
            print(f"  Faithfulness mean (nv): {faithfulness:.4f} (non-zero count: {len(non_zero_values)}/{len(df)})")
        elif "response_faithfulness" in df.columns:
            non_zero_values = df["response_faithfulness"][df["response_faithfulness"] > 0.001]
            faithfulness = float(non_zero_values.mean()) if len(non_zero_values) > 0 else 0.0
            print(f"  Faithfulness mean (response): {faithfulness:.4f} (non-zero count: {len(non_zero_values)}/{len(df)})")
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
        
        # Compute on all samples (relevancy doesn't need GT or contexts)
        all_samples_dataset = Dataset.from_list([
            {
                "question": sample['question'],
                "answer": sample["answer"],
                "contexts": sample.get("contexts", []),
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
        
        # Extract relevancy score (excluding zeros from average)
        if "answer_relevancy" in relevancy_df.columns:
            relevancy_values = relevancy_df["answer_relevancy"].tolist()
            non_zero_values = relevancy_df["answer_relevancy"][relevancy_df["answer_relevancy"] > 0.001]
            relevancy = float(non_zero_values.mean()) if len(non_zero_values) > 0 else 0.0
            print(f"  AnswerRelevancy values: {relevancy_values}")
            print(f"  AnswerRelevancy mean (non-zero only): {relevancy:.4f}")
            print(f"  Non-zero count: {len(non_zero_values)}/{len(relevancy_values)}")
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
                    "contexts": sample.get("contexts", []),
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
                non_zero_values = correctness_df["answer_correctness"][correctness_df["answer_correctness"] > 0.001]
                correctness = float(non_zero_values.mean()) if len(non_zero_values) > 0 else 0.0
                print(f"  AnswerCorrectness mean (non-zero only): {correctness:.4f}")
                print(f"  Non-zero count: {len(non_zero_values)}/{len(correctness_df)}")
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
