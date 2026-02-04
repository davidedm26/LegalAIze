import os
import json
import yaml
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    vect_params = params['vectorization']
    ingest_params = params['ingestion']
    
    processed_dir = ingest_params['processed_data_dir']
    chunks_path = os.path.join(processed_dir, "chunks.json")
    
    if not os.path.exists(chunks_path):
        print(f"⚠ File {chunks_path} non trovato. Esegui prima lo step di ingestion.")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Caricamento modello di embedding: {vect_params['model_name']}...")
    model = SentenceTransformer(vect_params['model_name'])
    
    # Inizializza client Qdrant (usiamo storage locale per integrazione DVC)
    # Se qdrant_host è localhost, proviamo a connetterci, altrimenti usiamo path locale
    vector_index_path = vect_params['vector_index_path']
    if not os.path.exists(vector_index_path):
        os.makedirs(vector_index_path)
    
    print(f"Inizializzazione Qdrant in {vector_index_path}...")
    client = QdrantClient(path=vector_index_path)
    
    collection_name = vect_params['collection_name']
    
    # Crea collezione
    vector_size = model.get_sentence_embedding_dimension()
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    
    print(f"Generazione embeddings per {len(chunks)} chunk...")
    
    points = []
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk['content']).tolist()
        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload={
                "source": chunk['source'],
                "content": chunk['content'],
                "chunk_id": chunk['chunk_id']
            }
        ))
        
    # Upsert in batch
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    # Creiamo un file di marker per DVC
    with open(os.path.join(vector_index_path, "status.json"), "w") as f:
        json.dump({"status": "indexed", "count": len(chunks)}, f)
        
    print(f"✓ Vectorization completata. Index salvato in {vector_index_path}")

if __name__ == "__main__":
    main()
