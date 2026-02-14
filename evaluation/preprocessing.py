"""Preprocessing utilities for evaluation."""

from typing import List, Dict, Any, Optional
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_document_for_groundedness(
    document_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Split document into chunks for groundedness evaluation."""
    if not document_text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(document_text)


def build_requirement_question(mapped_id: Optional[str], requirement_name: Optional[str]) -> str:
    """Build a question text (For RAGAS) for a specific requirement """
    identifier = mapped_id or "Unknown requirement"
    title = requirement_name or identifier
    # Only return title and id, no metadata
    return f"{title} ({identifier})"


def select_relevant_contexts(
    reference_text: str,
    chunk_texts: List[str],
    chunk_embeddings: Optional[np.ndarray],
    chunk_norms: Optional[np.ndarray],
    top_k: int,
    embedding_model=None,
) -> List[str]:
    """Select the most relevant document chunks based on similarity to reference text.
    embedding_model must be passed as a parameter.
    """
    if not chunk_texts:
        return []
    if top_k <= 0:
        return []
    if (
        not reference_text
        or chunk_embeddings is None
        or chunk_norms is None
        or embedding_model is None
    ):
        return chunk_texts[:top_k]

    note_vec = embedding_model.encode(
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
    """Extract ground truth auditor notes from a CSV row."""
    return (
        row.get("Auditor Notes")
        or row.get("auditor_notes")
        or row.get("Auditor_Notes")
    )
