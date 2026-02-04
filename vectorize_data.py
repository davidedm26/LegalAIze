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

    print(f"Loading embedding model: {vect_params['model_name']}...") # Load embedding model
    model = SentenceTransformer(vect_params['model_name']) # Load sentence transformer model
    
    # Initialize Qdrant client
    vector_index_path = vect_params['vector_index_path'] # Path for vector index (local for DVC compatibility)
    if not os.path.exists(vector_index_path): # Create directory if it doesn't exist
        os.makedirs(vector_index_path) # Create directory for vector index if it doesn't exist
    
    print(f"Initializing Qdrant at {vector_index_path}...")
    client = QdrantClient(path=vector_index_path) # Initialize Qdrant client 
    
    collection_name = vect_params['collection_name'] # Name of the collection in Qdrant
    
    vector_size = model.get_sentence_embedding_dimension() # Get embedding dimension from model
    client.recreate_collection( # Create or recreate collection in Qdrant
        collection_name=collection_name, # Collection name from params
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE), # Vector configuration with size and distance metric
    )
    
    print(f"Generating embeddings for {len(chunks)} chunks...")
    
    points = [] # List to hold points to be upserted in Qdrant
    for i, chunk in enumerate(chunks): # For each chunk, generate embedding and create PointStruct
        embedding = model.encode(chunk['content']).tolist() # Generate embedding for chunk content
        points.append(PointStruct( 
            id=i, # Unique ID for the point
            vector=embedding, # Embedding vector
            payload={ # Payload with metadata
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
    
    # Save indexing status
    with open(os.path.join(vector_index_path, "status.json"), "w") as f:
        json.dump({"status": "indexed", "count": len(chunks)}, f)
        
    print(f"✓ Vectorization completed. Index saved in {vector_index_path}")

if __name__ == "__main__":
    main()
