import os
import yaml
import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI

load_dotenv() # Load environment variables from .env file

@asynccontextmanager # Lifespan event to initialize RAG components, it runs on startup
async def lifespan(app: FastAPI):
    init_rag() # Initialize RAG components
    yield # Yield control back to FastAPI

app = FastAPI(title="LegalAIze Audit API", version="1.0.0", lifespan=lifespan) # FastAPI instance

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
mapping = None  # Mapping of requirements
llm = None  # LLM for evaluation

def init_rag(): # Initialize RAG components
    global embedding_model, vector_db, mapping, llm
    try:
        model_name = vect_params.get('model_name', "all-MiniLM-L6-v2") # Model name
        embedding_model = SentenceTransformer(model_name) # Load embedding model
        
        index_path = vect_params.get('vector_index_path', "data/processed/vector_index") # Vector index path
        search_paths = [index_path, os.path.join("..", index_path), os.path.abspath(index_path)] # Possible search paths
        
        final_path = next((p for p in search_paths if os.path.exists(p)), None)
        if final_path:
            vector_db = QdrantClient(path=final_path) # Load Qdrant vector database
            print(f"✓ RAG Initialized with index: {final_path}")
        else:
            print("⚠ Vector index not found!")
        
        # Load mapping
        mapping_path = os.path.join("data", "mapping.txt")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
            print(f"✓ Mapping loaded with {len(mapping)} requirements")
        else:
            print("⚠ Mapping file not found!")
        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        print("✓ LLM Initialized")
    except Exception as e:
        print(f"⚠ Init Error: {e}")

# Pydantic models for request and response
# Model for individual search result
class SearchResult(BaseModel): 
    content: str 
    source: str
    score: float

# Model for requirement report
class RequirementReport(BaseModel):
    id: str
    name: str
    compliance_score: float
    findings: list[SearchResult]
    iso_ref: str
    ai_act_articles: list[dict]

# Model for audit response
class AuditResponse(BaseModel):
    overall_compliance_score: float
    requirements: list[RequirementReport]
    recommendations: str

@app.get("/")
def root():
    return {"status": "ok", "service": "LegalAIze Audit Tool", "rag_ready": vector_db is not None}

@app.post("/audit", response_model=AuditResponse)
async def audit(document_text: str = Body(..., embed=True)):
    """
    Verifica la conformità di un testo rispetto alle normative indicizzate.
    """
    if vector_db is None or embedding_model is None or mapping is None or llm is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        requirements_reports = []
        all_scores = []
        
        for req_name, req_data in mapping.items():
            # Crea embedding del testo del requisito
            req_text = req_data.get("iso_control_text", "") + " " + " ".join([art.get("text", "") for art in req_data.get("ai_act_articles", [])])
            req_vector = embedding_model.encode(req_text).tolist()
            
            # Cerca riferimenti normativi per questo requisito
            search_result = []
            if hasattr(vector_db, "query_points"):
                response = vector_db.query_points(
                    collection_name="legal_docs",
                    query=req_vector,
                    limit=3  # Limita per requisito
                )
                search_result = response.points
            elif hasattr(vector_db, "search"):
                search_result = vector_db.search(
                    collection_name="legal_docs",
                    query_vector=req_vector,
                    limit=3
                )
            elif hasattr(vector_db, "query"):
                try:
                    search_result = vector_db.query(
                        collection_name="legal_docs",
                        query_vector=req_vector,
                        limit=3
                    )
                except:
                    search_result = vector_db.query(
                        collection_name="legal_docs",
                        query_text=req_text,
                        limit=3
                    )
            
            findings = []
            chunks_text = []
            for hit in search_result:
                payload = getattr(hit, "payload", {}) if hasattr(hit, "payload") else hit.get("payload", {})
                score = getattr(hit, "score", 0) if hasattr(hit, "score") else hit.get("score", 0)
                
                content = payload.get("text", payload.get("content", "Testo non disponibile"))
                source = payload.get("source", "Unknown")
                
                findings.append(SearchResult(
                    content=content,
                    source=source,
                    score=score
                ))
                chunks_text.append(content)
            
            # Usa LLM per valutare la conformità
            prompt = f"""
            Valuta la conformità del seguente documento rispetto al requisito specificato.
            
            Documento da valutare:
            {document_text}
            
            Requisito:
            {req_text}
            
            Riferimenti normativi rilevanti (chunk estratti):
            {"\n".join(chunks_text)}
            
            Fornisci un punteggio di conformità da 0.0 a 1.0 (dove 1.0 significa pienamente conforme) e una breve spiegazione.
            Formato risposta: Punteggio: [numero], Spiegazione: [testo]
            """
            
            llm_response = llm.invoke(prompt).content.strip()
            
            # Estrai punteggio dalla risposta (assumi formato "Punteggio: 0.8, Spiegazione: ...")
            try:
                score_part = llm_response.split("Punteggio:")[1].split(",")[0].strip()
                req_score = float(score_part)
            except:
                req_score = 0.0  # Fallback
            
            all_scores.append(req_score)
            
            requirements_reports.append(RequirementReport(
                id=req_data["id"],
                name=req_name,
                compliance_score=round(req_score, 2),
                findings=findings,
                iso_ref=req_data["iso_ref"],
                ai_act_articles=req_data["ai_act_articles"]
            ))
        
        # Punteggio complessivo
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
        compliance_level = "HIGH" if overall_score > 0.7 else "MEDIUM" if overall_score > 0.5 else "LOW"
        
        return AuditResponse(
            overall_compliance_score=round(overall_score, 2),
            requirements=requirements_reports,
            recommendations=f"Analisi completata. Livello di corrispondenza normativa: {compliance_level}. Verificare i dettagli per ciascun requisito."
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
 