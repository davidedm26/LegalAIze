import os
import yaml
import json
from typing import Union
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
    with open("..\params.yaml", "r") as f:
        return yaml.safe_load(f)

params = load_params() # Load parameters
vect_params = params.get('vectorization', {}) # Vectorization parameters
eval_params = params.get('evaluation', {}) # Evaluation / LLM parameters

# Global variables for RAG components
embedding_model = None # Renamed to avoid conflicts with the module
vector_db = None  # Renamed to avoid conflicts with the module
mapping = None  # Mapping of requirements
llm = None  # LLM for evaluation
requirement_chunks = {}  # Pre-computed relevant chunks per requirement

def init_rag(): # Initialize RAG components
    global embedding_model, vector_db, mapping, llm, requirement_chunks
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
        mapping_path = os.path.join("..","data", "mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
            print(f"✓ Mapping loaded with {len(mapping)} requirements")
        else:
            print("⚠ Mapping file not found!")
        # Load pre-computed requirement chunks
        chunks_path_candidates = [
            os.path.join("..", "data", "processed", "requirement_chunks.json"),
            os.path.join("data", "processed", "requirement_chunks.json"),
        ]
        chunks_path = next((p for p in chunks_path_candidates if os.path.exists(p)), None)
        if chunks_path:
            with open(chunks_path, "r", encoding="utf-8") as f:
                requirement_chunks = json.load(f)
            print(f"✓ Requirement chunks loaded from {chunks_path}")
        else:
            print("⚠ Requirement chunks file not found! Run DVC stage 'precompute_rag'.")
        
        # Initialize LLM using configuration parameters
        llm_model_name = eval_params.get("llm_model")
        llm_temperature = float(eval_params.get("llm_temperature"))
        llm = ChatOpenAI(model=llm_model_name, temperature=llm_temperature, request_timeout=30)
        print(f"✓ LLM Initialized ({llm_model_name}, temp={llm_temperature})")
    except Exception as e:
        print(f"⚠ Init Error: {e}")

# Pydantic models for request and response

# Model for search result (chunk)
class SearchResult(BaseModel):  # Chunk found in vector DB
    content: str # Text content of the chunk
    source: str # Source document of the chunk
    score: float # Similarity score with the query

# Model for requirement report 
class RequirementReport(BaseModel): # Report for each requirement
    Mapped_ID: str # Requirement ID
    Requirement_Name: str # Requirement Name
    Score: Union[int, str] # Score from 0 to 5 or 'N/A'
    Auditor_Notes: str # Notes from LLM evaluation
    

# Model for audit response
class AuditResponse(BaseModel): 
    requirements: list[RequirementReport] # List of requirement reports

@app.get("/health") # Health check endpoint
def health(): 
    rag_ready = all([
        vector_db is not None,
        embedding_model is not None,
        mapping is not None,
        llm is not None,
        bool(requirement_chunks),
    ])
    return {"status": "healthy", "rag_ready": rag_ready}


@app.post("/audit", response_model=AuditResponse)
async def audit(document_text: str = Body(..., embed=True)): # Audit endpoint
    """
    Produce an audit report for the given document text.
    Takes the document text as input and returns the audit report.
    """
    if vector_db is None or embedding_model is None or mapping is None or llm is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        requirements_reports = [] # To store reports for each requirement
        all_scores = [] # To store all compliance scores
        
        limited_mapping = dict(list(mapping.items())[:20]) # Limit to first 20 requirements for performance
        
        for req_name, req_data in limited_mapping.items(): # For each requirement
            
            # Build full requirement text
            req_text = req_data.get("iso_control_text", "") + " " + " ".join([art.get("text", "") for art in req_data.get("ai_act_articles", [])]) # Full requirement text

            # Use precomputed normative chunks for this requirement
            pre_chunks = requirement_chunks.get(req_name, [])
            findings = [
                SearchResult(
                    content=ch.get("content", "Testo non disponibile"),
                    source=ch.get("source", "Unknown"),
                    score=float(ch.get("score", 0.0)),
                )
                for ch in pre_chunks
            ]
            chunks_text = [f.content for f in findings]
            
            # Use LLM to evaluate compliance
            chunks_joined = "\n".join(chunks_text) # Create a single string with all chunk texts
            prompt = f"""
You are an expert auditor in regulatory compliance for AI systems, specialized in AI Act and ISO/IEC 42001 standards.

Evaluate the compliance of the provided document against the specified requirement. Use the extracted regulatory references as additional context to interpret the requirement.

Document to evaluate:
{document_text}

Requirement to verify:
{req_text}

Relevant regulatory references (extracted from legal corpus):
{chunks_joined}

Instructions:
- Score: An integer from 0 to 5 (0 = no compliance, 5 = maximum compliance).
- Auditor Notes: A concise note (max 100 words) explaining the evaluation, citing evidence from the document and references.
- If you cannot determine a score, return N/A for the score and explain in the notes.
Respond exclusively in valid JSON format:
{{
    "score": integer from 0 to 5,
  "auditor_notes": "note text"
}}
"""
            
            llm_response = llm.invoke(prompt).content.strip() # Get LLM response
            
            # Parse LLM response
            try:
                # Remove markdown code blocks if present
                cleaned_response = llm_response.replace("```json", "").replace("```", "").strip()
                response_json = json.loads(cleaned_response)
                    
                score_0_5 = int(response_json["score"])
                auditor_notes = response_json["auditor_notes"]
                req_score = score_0_5 / 5.0
            except:
                score_0_5 = 0
                auditor_notes = "LLM response parsing failed. Response was: " + llm_response
                req_score = 0.0
            all_scores.append(req_score)
            
            requirements_reports.append(RequirementReport(
                Mapped_ID=req_data["id"],
                Requirement_Name=req_name,
                Score=score_0_5 if isinstance(score_0_5, int) else "N/A",
                Auditor_Notes=auditor_notes
            ))
        
        # Store the AuditResponse before returning, for potential future use (e.g., in-memory cache, database, etc.)
        debug_dir_path = os.path.join("..", "data", "debug")
        os.makedirs(debug_dir_path, exist_ok=True)
        with open(os.path.join(debug_dir_path, "audit_report_example.json"), "w", encoding="utf-8") as f:
            json.dump([r.dict() for r in requirements_reports], f, ensure_ascii=False, indent=4)
        
        # Return the report as a list of requirements
        return AuditResponse(
            requirements=requirements_reports
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit failed: {str(e)}")



@app.get("/model_info")
def model_info():
    """Comprehensive information about the system components"""
    info = {
        "embedding_model": {
            "loaded": embedding_model is not None,
            "type": type(embedding_model).__name__ if embedding_model else None,
            "model_name": getattr(embedding_model, 'model_name', None) if embedding_model else None,
            "max_seq_length": getattr(embedding_model, 'max_seq_length', None) if embedding_model else None,
        },
        "vector_db": {
            "loaded": vector_db is not None,
            "type": type(vector_db).__name__ if vector_db else None,
            "path": getattr(vector_db, '_location', None) if vector_db else None,
        },
        "llm": {
            "loaded": llm is not None,
            "model_name": getattr(llm, 'model_name', None) if llm else None,
            "provider": "OpenAI" if llm else None,
        },
        "mapping": {
            "loaded": mapping is not None,
            "num_requirements": len(mapping) if mapping else 0,
        }
    }
    return info
 