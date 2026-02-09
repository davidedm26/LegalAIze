"""Metrics computation for RAG evaluation."""

from typing import List, Dict, Any, Optional
import numpy as np

try:
    from datasets import Dataset
except ImportError:
    Dataset = None

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import Faithfulness, ResponseGroundedness
except ImportError:
    ragas_evaluate = None
    ResponseGroundedness = None
    Faithfulness = None

RAGAS_GROUNDEDNESS_AVAILABLE = (
    Dataset is not None
    and ragas_evaluate is not None
    and ResponseGroundedness is not None
)
RAGAS_FAITHFULNESS_AVAILABLE = (
    Dataset is not None
    and ragas_evaluate is not None
    and Faithfulness is not None
)


def compute_mae(gt_scores: List[float], pred_scores: List[float]) -> float:
    """Compute Mean Absolute Error between ground truth and predicted scores."""
    if not gt_scores:
        return 0.0
    diffs = [abs(g - p) for g, p in zip(gt_scores, pred_scores)]
    return sum(diffs) / len(diffs)


def compute_note_similarity(gt_note: str, pred_note: str) -> float:
    """Compute cosine similarity between ground-truth and predicted notes using embeddings."""
    if not gt_note or not pred_note:
        return 0.0
    
    try:
        from backend import rag_engine
    except ImportError:
        raise RuntimeError("backend.rag_engine is not available.")
    
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


def compute_groundedness_score(samples: List[Dict[str, Any]]) -> Optional[float]:
    """Compute groundedness score for a list of samples using Ragas."""
    if not samples:
        return None
    if not RAGAS_GROUNDEDNESS_AVAILABLE:
        print("⚠ RAGAS groundedness metric unavailable (check ragas/datasets installation).")
        return None
    
    try:
        from backend import rag_engine
    except ImportError:
        return None
    
    try:
        assert Dataset is not None
        assert ragas_evaluate is not None
        assert ResponseGroundedness is not None
        assert rag_engine is not None

        ragas_dataset = Dataset.from_list(
            [
                {
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "contexts": sample["contexts"],
                    "ground_truth": sample.get("ground_truth", ""),
                }
                for sample in samples
            ]
        )
        ragas_result = ragas_evaluate(
            dataset=ragas_dataset,
            metrics=[ResponseGroundedness()],
            llm=rag_engine.llm,
        )
        # ragas_result is an EvaluationResult - use to_pandas() to get scores
        df = ragas_result.to_pandas()
        # Look for nv_response_groundedness column (NV = NVIDIA variant)
        if 'nv_response_groundedness' in df.columns:
            return float(df['nv_response_groundedness'].mean())
        elif 'response_groundedness' in df.columns:
            return float(df['response_groundedness'].mean())
        elif 'groundedness' in df.columns:
            return float(df['groundedness'].mean())
        else:
            print(f"⚠ Available score columns: {list(df.columns)}")
            return None
    except Exception as e:
        print(f"⚠ Failed to compute groundedness: {e}")
        return None


def compute_faithfulness_score(samples: List[Dict[str, Any]]) -> Optional[float]:
    """Compute faithfulness score for a list of samples using Ragas."""
    if not samples:
        return None
    if not RAGAS_FAITHFULNESS_AVAILABLE:
        print("⚠ RAGAS faithfulness metric unavailable (check ragas/datasets installation).")
        return None

    try:
        from backend import rag_engine
    except ImportError:
        return None

    try:
        assert Dataset is not None
        assert ragas_evaluate is not None
        assert Faithfulness is not None
        assert rag_engine is not None

        ragas_dataset = Dataset.from_list(
            [
                {
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "contexts": sample["contexts"],
                    "ground_truth": sample.get("ground_truth", ""),
                }
                for sample in samples
            ]
        )
        ragas_result = ragas_evaluate(
            dataset=ragas_dataset,
            metrics=[Faithfulness()],
            llm=rag_engine.llm,
        )
        df = ragas_result.to_pandas()
        if "nv_response_faithfulness" in df.columns:
            return float(df["nv_response_faithfulness"].mean())
        elif "response_faithfulness" in df.columns:
            return float(df["response_faithfulness"].mean())
        elif "faithfulness" in df.columns:
            return float(df["faithfulness"].mean())
        else:
            print(f"⚠ Available faithfulness columns: {list(df.columns)}")
            return None
    except Exception as e:
        print(f"⚠ Failed to compute faithfulness: {e}")
        return None
