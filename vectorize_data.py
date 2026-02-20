"""   
Chunks embedding and indexing script. This script takes the processed regulatory chunks, generates vector embeddings using a specified model, and indexes them into a Qdrant vector database for efficient retrieval during RAG operations. It supports both local Qdrant instances (using file-based storage) and remote Qdrant services (via host/port connection), allowing flexibility for different deployment scenarios.
"""
import os
import json
import yaml

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import requests
import time
from typing import List 


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)
    

def main():
    params = load_params() # Load parameters from params.yaml
    vect_params = params['vectorization'] # Vectorization parameters
    ingest_params = params['ingestion'] # Ingestion parameters
    
    processed_dir = ingest_params['processed_data_dir'] # Directory for processed data
    chunks_path = os.path.join(processed_dir, "requirement_chunks.json") # Path to chunks file
    
    if not os.path.exists(chunks_path): # Check if chunks file exists
        print(f"⚠ File {chunks_path} not found. Run the ingestion step first.")
        return

    with open(chunks_path, "r", encoding="utf-8") as f: # Load chunks from JSON file
        chunks = json.load(f)
    

    # Decide mode: local qdrant process (path) or remote service (host:port)
    # If the QDRANT_HOST environment variable is set, we assume remote service mode; otherwise, we use local path-based storage. This allows flexibility for different deployment scenarios (local development vs production).

    qdrant_host = os.getenv("QDRANT_HOST") # Use localhost to connect to qdrant container when running this script in local environment.
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    vector_index_path = vect_params.get('vector_index_path')
    collection_name = vect_params.get('collection_name')

    # Helper function to wait for Qdrant service to be ready (only relevant for remote service mode)
    def wait_for_qdrant(host: str, port: int, timeout: int = 60):
        url = f"http://{host}:{port}/healthz"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False

    if qdrant_host: # If QDRANT_HOST is set, we're in remote service mode
        print(f"Using remote Qdrant service at {qdrant_host}:{qdrant_port}")
        if not wait_for_qdrant(qdrant_host, qdrant_port, timeout=60):
            print("⚠ Qdrant service not ready (health check failed). Aborting.")
            return
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
    else:
        # Embedded Qdrant client with local storage 
        if not vector_index_path:
            print("⚠ Missing 'vector_index_path' in params for local Qdrant storage.")
            return
        if not os.path.exists(vector_index_path):
            os.makedirs(vector_index_path)
        print(f"Initializing local Qdrant at {vector_index_path}...")
        client = QdrantClient(path=vector_index_path)


    print(f"Loading embedding model: {vect_params['model_name']}...")
    model = SentenceTransformer(vect_params['model_name'])
    if model is None:
        print(f"⚠ Failed to load embedding model '{vect_params['model_name']}'. Check model name and availability.")
        return
    # Ensure collection exists 
    vector_size = model.get_sentence_embedding_dimension() # Get vector size from model
    client.recreate_collection( 
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    print(f"Generating embeddings for {len(chunks)} chunks...")

    flat_chunks = []
    for req in chunks:
        principle = req.get("ethicalPrinciple")
        req_name = req.get("requirementName")
        id = req.get("id", "")
        # EU AI Act articles
        for art in req.get("euAiActArticles", []):
            flat_chunks.append({
                "source": "EU_AI_ACT",
                "ethicalPrinciple": principle,
                "requirementId": id,
                "requirementName": req_name,
                "reference": art.get("reference"),
                "content": art.get("content"),
            })
        # ISO references
        for iso in req.get("iso42001Reference", []):
            flat_chunks.append({
                "source": "ISO_42001",
                "ethicalPrinciple": principle,
                "requirementId": id,
                "requirementName": req_name,
                "reference": iso.get("reference"),
                "content": iso.get("content"),
            })

    print(f"Generating embeddings for {len(flat_chunks)} atomic chunks...")

    texts = [c.get("content", "") for c in flat_chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    batch_size = vect_params.get('batch_size', 128)
    points: List[PointStruct] = []
    for i, chunk in enumerate(flat_chunks):
        emb = embeddings[i].tolist()
        points.append(PointStruct(id=i, vector=emb, payload={
            "source": chunk.get('source'),
            "ethicalPrinciple": chunk.get('ethicalPrinciple'),
            "requirementId": chunk.get('requirementId'),
            "requirementName": chunk.get('requirementName'),
            "reference": chunk.get('reference'),
            "content": chunk.get('content'),
        }))

    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        client.upsert(collection_name=collection_name, points=batch_points)

    # Save indexing status locally if vector_index_path present
    if vector_index_path:
        status_path = os.path.join(vector_index_path, "status.json")
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump({"status": "indexed", "count": len(chunks)}, f)
        
    else:
        print("Note: no local vector_index_path configured; skipping snapshot creation for remote Qdrant.")

    print("✓ Vectorization completed.")

if __name__ == "__main__":
    main()
