import os
import json
import yaml
from typing import Dict, List, Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


def load_params() -> Dict[str, Any]:
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mapping(mapping_path: str) -> Dict[str, Any]:
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found at {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_requirement_text(req_data: Dict[str, Any]) -> str:
    iso_text = req_data.get("iso_control_text", "")
    articles = " ".join([art.get("text", "") for art in req_data.get("ai_act_articles", [])])
    return f"{iso_text} {articles}".strip()


def main() -> None:
    params = load_params()
    vect_params = params["vectorization"]
    ingest_params = params["ingestion"]

    processed_dir = ingest_params["processed_data_dir"]  # e.g. data/processed
    os.makedirs(processed_dir, exist_ok=True)

    # Load mapping.json
    mapping_path = os.path.join("data", "mapping.json")
    mapping = load_mapping(mapping_path)
    print(f"Loaded mapping with {len(mapping)} requirements from {mapping_path}")

    # Load embedding model (same as vectorize step)
    model_name = vect_params["model_name"]
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)

    # Init Qdrant (same path as vectorize)
    vector_index_path = vect_params["vector_index_path"]  # e.g. data/processed/vector_index
    if not os.path.exists(vector_index_path):
        raise FileNotFoundError(
            f"Vector index path {vector_index_path} not found. Run 'vectorize' stage first."
        )

    print(f"Connecting to Qdrant at {vector_index_path} ...")
    client = QdrantClient(path=vector_index_path)
    collection_name = vect_params["collection_name"]

    top_k = int(vect_params.get("top_k", 3))
    print(f"Using collection '{collection_name}' with top_k={top_k} per requirement")

    requirement_embeddings: Dict[str, List[float]] = {}
    requirement_chunks: Dict[str, List[Dict[str, Any]]] = {}

    for req_name, req_data in mapping.items():
        req_id = req_data.get("id", req_name)
        print(f"Processing requirement: {req_id} | {req_name}")

        # Build full requirement text and encode
        req_text = build_requirement_text(req_data)
        req_vector = model.encode(req_text).tolist()
        requirement_embeddings[req_name] = req_vector

        # Query Qdrant for normative chunks
        response = client.query_points(
            collection_name=collection_name,
            query=req_vector,
            limit=top_k,
        )

        chunks_for_req: List[Dict[str, Any]] = []
        for hit in response.points:
            # hit can be a pydantic-like object or dict depending on client version
            payload = getattr(hit, "payload", None) or getattr(hit, "dict", lambda: {})().get("payload", {})
            if isinstance(payload, dict) is False:
                payload = {}

            score = getattr(hit, "score", None)
            if score is None and hasattr(hit, "dict"):
                score = hit.dict().get("score", 0.0)

            content = payload.get("content") or payload.get("text") or "Testo non disponibile"
            source = payload.get("source", "Unknown")
            chunk_id = payload.get("chunk_id")

            chunks_for_req.append(
                {
                    "content": content,
                    "source": source,
                    "score": float(score) if score is not None else 0.0,
                    "chunk_id": chunk_id,
                }
            )

        requirement_chunks[req_name] = chunks_for_req

    # Save embeddings
    embeddings_path = os.path.join(processed_dir, "requirement_embeddings.json")
    with open(embeddings_path, "w", encoding="utf-8") as f:
        json.dump(requirement_embeddings, f)
    print(f"Saved requirement embeddings to {embeddings_path}")

    # Save chunks
    chunks_path = os.path.join(processed_dir, "requirement_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(requirement_chunks, f)
    print(f"Saved requirement chunks to {chunks_path}")

    print("✓ Precompute RAG step completed.")


if __name__ == "__main__":
    main()
