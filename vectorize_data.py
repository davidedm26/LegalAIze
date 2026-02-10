import os
import json
import yaml
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import requests
import time
from typing import List 
import shutil
from pathlib import Path

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params() # Load parameters from params.yaml
    vect_params = params['vectorization'] # Vectorization parameters
    ingest_params = params['ingestion'] # Ingestion parameters
    
    processed_dir = ingest_params['processed_data_dir'] # Directory for processed data
    chunks_path = os.path.join(processed_dir, "chunks.json") # Path to chunks file
    
    if not os.path.exists(chunks_path): # Check if chunks file exists
        print(f"⚠ File {chunks_path} not found. Run the ingestion step first.")
        return

    with open(chunks_path, "r", encoding="utf-8") as f: # Load chunks from JSON file
        chunks = json.load(f)

    print(f"Loading embedding model: {vect_params['model_name']}...")
    model = SentenceTransformer(vect_params['model_name'])

    # Decide mode: local qdrant process (path) or remote service (host:port)
    qdrant_host = os.getenv("QDRANT_HOST") # Use localhost to connect to qdrant container when running this script in local environment.
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    vector_index_path = vect_params.get('vector_index_path')
    collection_name = vect_params.get('collection_name')

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

    if qdrant_host:
        print(f"Using remote Qdrant service at {qdrant_host}:{qdrant_port}")
        if not wait_for_qdrant(qdrant_host, qdrant_port, timeout=60):
            print("⚠ Qdrant service not ready (health check failed). Aborting.")
            return
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
    else:
        # local path-based qdrant
        if not vector_index_path:
            print("⚠ Missing 'vector_index_path' in params for local Qdrant storage.")
            return
        if not os.path.exists(vector_index_path):
            os.makedirs(vector_index_path)
        print(f"Initializing local Qdrant at {vector_index_path}...")
        client = QdrantClient(path=vector_index_path)

    # Ensure collection exists (recreate to keep idempotent)
    vector_size = model.get_sentence_embedding_dimension()
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    print(f"Generating embeddings for {len(chunks)} chunks...")

    texts = [c.get("content", "") for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Upsert in batches to avoid very large requests
    batch_size = vect_params.get('batch_size', 128)
    points: List[PointStruct] = []
    for i, chunk in enumerate(chunks):
        emb = embeddings[i].tolist()
        points.append(PointStruct(id=i, vector=emb, payload={
            "source": chunk.get('source'),
            "content": chunk.get('content'),
            "chunk_id": chunk.get('chunk_id')
        }))

    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        client.upsert(collection_name=collection_name, points=batch_points)

    # Save indexing status locally if vector_index_path present
    if vector_index_path:
        status_path = os.path.join(vector_index_path, "status.json")
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump({"status": "indexed", "count": len(chunks)}, f)
        '''
        # Optionally create a portable snapshot copy of the vector_index for DVC
        snapshot_path = vect_params.get("snapshot_path") or os.path.join(os.path.dirname(vector_index_path), "vector_index_snapshot")
        try:
            print(f"Requesting Qdrant server snapshot for collection '{collection_name}'...")
            snapshot_resp = None
            try:
                # Prefer explicit collection_name when supported
                snapshot_resp = client.create_snapshot(collection_name=collection_name, wait=True)
            except TypeError:
                # Some client versions may not accept collection_name; try without it
                snapshot_resp = client.create_snapshot(wait=True)
            except Exception as e:
                print(f"⚠ Qdrant create_snapshot() failed: {e}")

            # Close client (release handles). We won't write local snapshot_meta.json per request.
            try:
                client.close()
            except Exception:
                pass

            print("✓ Snapshot requested on Qdrant server (no local metadata written)")
        except Exception as e:
            print(f"⚠ Failed to request/create snapshot: {e}")
        '''
    else:
        print("Note: no local vector_index_path configured; skipping snapshot creation for remote Qdrant.")

    print("✓ Vectorization completed.")

if __name__ == "__main__":
    main()
