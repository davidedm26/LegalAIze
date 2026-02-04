import os
import yaml
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

app = FastAPI(title="LegalAIze Audit API", version="1.0.0") # FastAPI instance

# CORS - Cross-Origin Resource Sharing , allow all origins for simplicity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load parameters from params.yaml
def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

params = load_params() # Load parameters
vect_params = params.get('vectorization', {}) # Vectorization parameters

# Global variables for RAG components
embedding_model = None # Renamed to avoid conflicts with the module
vector_db = None  # Renamed to avoid conflicts with the module

def init_rag(): # Initialize RAG components
    global embedding_model, vector_db
    try:
        model_name = vect_params.get('model_name', "all-MiniLM-L6-v2") # Model name
        embedding_model = SentenceTransformer(model_name) # Load embedding model
        
        index_path = vect_params.get('vector_index_path', "data/processed/vector_index")
        search_paths = [index_path, os.path.join("..", index_path), os.path.abspath(index_path)]
        
        final_path = next((p for p in search_paths if os.path.exists(p)), None)
        if final_path:
            # Creiamo l'istanza esplicitamente
            vector_db = QdrantClient(path=final_path)
            print(f"✓ RAG Initialized with index: {final_path}")
        else:
            print("⚠ Vector index not found!")
    except Exception as e:
        print(f"⚠ Init Error: {e}")

@app.on_event("startup")
async def startup_event():
    init_rag()

class SearchResult(BaseModel):
    content: str
    source: str
    score: float

class AuditResponse(BaseModel):
    compliance_score: float
    findings: list[SearchResult]
    recommendations: str

@app.get("/")
def root():
    return {"status": "ok", "service": "LegalAIze Audit Tool", "rag_ready": vector_db is not None}

@app.post("/audit", response_model=AuditResponse)
async def audit(document_text: str = Body(..., embed=True)):
    """
    Verifica la conformità di un testo rispetto alle normative indicizzate.
    """
    if vector_db is None or embedding_model is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # 1. Crea embedding del documento in input
        vector = embedding_model.encode(document_text).tolist()
        
        # 2. Cerca riferimenti normativi pertinenti (Supporto multiversione Qdrant)
        search_result = []
        if hasattr(vector_db, "query_points"):
            # Metodo moderno (Qdrant 1.10+) - raccomandato dato che 'search' è sparito
            response = vector_db.query_points(
                collection_name="legal_docs",
                query=vector,
                limit=5
            )
            search_result = response.points
        elif hasattr(vector_db, "search"):
            # Metodo classico
            search_result = vector_db.search(
                collection_name="legal_docs",
                query_vector=vector,
                limit=5
            )
        elif hasattr(vector_db, "query"):
            # Fallback per versioni specifiche con FastEmbed
            try:
                search_result = vector_db.query(
                    collection_name="legal_docs",
                    query_vector=vector,
                    limit=5
                )
            except:
                # Se proprio fallisce, usiamo la query testuale (ma abbiamo già il vettore)
                search_result = vector_db.query(
                    collection_name="legal_docs",
                    query_text=document_text,
                    limit=5
                )
        
        findings = []
        scores = []
        for hit in search_result:
            # Gestione versatile del payload e dello score
            payload = getattr(hit, "payload", {}) if hasattr(hit, "payload") else hit.get("payload", {})
            score = getattr(hit, "score", 0) if hasattr(hit, "score") else hit.get("score", 0)
            
            content = payload.get("text", payload.get("content", "Testo non disponibile"))
            source = payload.get("source", "Unknown")
            
            findings.append(SearchResult(
                content=content,
                source=source,
                score=score
            ))
            scores.append(score)
            
        # 3. Logica semplificata di audit
        avg_score = sum(scores) / len(scores) if scores else 0
        compliance_level = "HIGH" if avg_score > 0.7 else "MEDIUM" if avg_score > 0.5 else "LOW"
        
        return AuditResponse(
            compliance_score=round(avg_score, 2),
            findings=findings,
            recommendations=f"Analisi completata. Livello di corrispondenza normativa: {compliance_level}. Verificare i dettagli nei riferimenti trovati."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit failed: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "rag_ready": vector_db is not None}

@app.get("/model_info")
def model_info():
    """Informazioni sul modello di embedding"""
    if embedding_model is None:
        return {"error": "Embedding model not loaded"}
    return {
        "type": type(embedding_model).__name__,
        "model_name": getattr(embedding_model, 'model_name', str(embedding_model))
    }
 