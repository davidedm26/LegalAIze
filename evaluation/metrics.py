"""Metrics computation for RAG evaluation."""

from typing import List, Dict, Any, Optional
import numpy as np

try:
    from datasets import Dataset
except ImportError:
    Dataset = None



try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import Faithfulness, ResponseGroundedness, AnswerRelevancy, AnswerCorrectness
except ImportError:
    ragas_evaluate = None
    ResponseGroundedness = None
    Faithfulness = None
    AnswerRelevancy = None
    AnswerCorrectness = None



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


def compute_note_similarity(gt_note: str, pred_note: str, embedding_model) -> float:
    """
    Compute cosine similarity between ground-truth and predicted notes using embeddings.
    Args:
        gt_note (str): Ground-truth note text.
        pred_note (str): Predicted note text.
        embedding_model: SentenceTransformer or compatible model with .encode().
    Returns:
        float: Cosine similarity between the two notes in [-1, 1].
    """
    if not gt_note or not pred_note:
        return 0.0
    if embedding_model is None:
        raise ValueError("embedding_model must be provided to compute_note_similarity.")
    
    embeddings = embedding_model.encode(
        [gt_note, pred_note],
        convert_to_numpy=True,
    )
    gt_vec, pred_vec = embeddings
    gt_norm = np.linalg.norm(gt_vec)
    pred_norm = np.linalg.norm(pred_vec)
    if not gt_norm or not pred_norm:
        return 0.0
    similarity = float(np.dot(gt_vec, pred_vec) / (gt_norm * pred_norm)) # Cosine similarity (Dot product divided by norms)

    return max(min(similarity, 1.0), -1.0) # Ensure similarity is in [-1, 1] range





def compute_ragas_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Compute groundedness, faithfulness, relevancy, and correctness scores for a list of samples using a single Ragas evaluation call.
    Returns a dict with keys 'groundedness', 'faithfulness', 'relevancy', and 'correctness'.
    """
    if not samples:
        return {"groundedness": None, "faithfulness": None, "relevancy": None, "correctness": None}
    if not (RAGAS_GROUNDEDNESS_AVAILABLE and RAGAS_FAITHFULNESS_AVAILABLE and RAGAS_RELEVANCY_AVAILABLE and RAGAS_CORRECTNESS_AVAILABLE):
        print("⚠ RAGAS metrics unavailable (check ragas/datasets installation).")
        return {"groundedness": None, "faithfulness": None, "relevancy": None, "correctness": None}

    try:
        from backend import rag_engine
    except ImportError:
        return {"groundedness": None, "faithfulness": None, "relevancy": None, "correctness": None}

    try:
        assert Dataset is not None
        assert ragas_evaluate is not None
        assert ResponseGroundedness is not None
        assert Faithfulness is not None
        assert AnswerRelevancy is not None
        assert AnswerCorrectness is not None
        assert rag_engine is not None

        ragas_dataset = Dataset.from_list([
            {
                "question": f"Evaluate compliance for: {sample['question']}",
                "answer": sample["answer"],
                "contexts": sample["contexts"],
                "ground_truth": sample.get("ground_truth", ""),
            }
            for sample in samples
        ])
        ragas_result = ragas_evaluate(
            dataset=ragas_dataset,
            metrics=[ResponseGroundedness(), Faithfulness(), AnswerRelevancy(), AnswerCorrectness()],
            llm=rag_engine.llm,
        )
        df = ragas_result.to_pandas()
        # Extract groundedness
        if 'nv_response_groundedness' in df.columns:
            groundedness = float(df['nv_response_groundedness'].mean())
        elif 'response_groundedness' in df.columns:
            groundedness = float(df['response_groundedness'].mean())
        elif 'groundedness' in df.columns:
            groundedness = float(df['groundedness'].mean())
        else:
            groundedness = None
        # Extract faithfulness
        if "nv_response_faithfulness" in df.columns:
            faithfulness = float(df["nv_response_faithfulness"].mean())
        elif "response_faithfulness" in df.columns:
            faithfulness = float(df["response_faithfulness"].mean())
        elif "faithfulness" in df.columns:
            faithfulness = float(df["faithfulness"].mean())
        else:
            faithfulness = None
        # Extract relevance
        if "nv_response_answer_relevancy" in df.columns:
            relevancy = float(df["nv_response_answer_relevancy"].mean())
        elif "response_answer_relevancy" in df.columns:
            relevancy = float(df["response_answer_relevancy"].mean())
        elif "answer_relevancy" in df.columns:
            relevancy = float(df["answer_relevancy"].mean())
        else:
            relevancy = None
        # Extract correctness
        if "nv_response_answer_correctness" in df.columns:
            correctness = float(df["nv_response_answer_correctness"].mean())
        elif "response_answer_correctness" in df.columns:
            correctness = float(df["response_answer_correctness"].mean())
        elif "answer_correctness" in df.columns:
            correctness = float(df["answer_correctness"].mean())
        else:
            correctness = None
        return {"groundedness": groundedness, "faithfulness": faithfulness, "relevancy": relevancy, "correctness": correctness}
    except Exception as e:
        print(f"⚠ Failed to compute RAGAS metrics: {e}")
        return {"groundedness": None, "faithfulness": None, "relevancy": None, "correctness": None}
