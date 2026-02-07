import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

try:
    from . import rag_engine
    from .rag_engine import AuditResponse
except ImportError:  # Fallback when running as script (python app.py)
    import rag_engine  # type: ignore
    from rag_engine import AuditResponse  # type: ignore

load_dotenv() # Load environment variables from .env file

DEBUG_DUMP_PATH = os.path.join("..", "data", "debug", "audit_report_example.json")


@asynccontextmanager # Lifespan event to initialize RAG components, it runs on startup
async def lifespan(app: FastAPI):
    try:
        rag_engine.init_rag()
        yield
    finally:
        # No teardown logic required for now
        pass


app = FastAPI(title="LegalAIze Audit API", version="1.0.0", lifespan=lifespan) # FastAPI instance

# CORS - Cross-Origin Resource Sharing , allow all origins for simplicity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health") # Health check endpoint
def health():
    return {"status": "healthy", "rag_ready": rag_engine.rag_ready()}


@app.post("/audit", response_model=AuditResponse)
async def audit(document_text: str = Body(..., embed=True)): # Audit endpoint
    """
    Produce an audit report for the given document text.
    Takes the document text as input and returns the audit report.
    """
    if not rag_engine.rag_ready():
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        return rag_engine.audit_document(
            document_text,
            debug_dump_path=DEBUG_DUMP_PATH,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audit failed: {exc}")


@app.get("/model_info")
def model_info():
    """Comprehensive information about the system components"""
    embedding_model = rag_engine.embedding_model
    vector_db = rag_engine.vector_db
    llm = rag_engine.llm
    mapping = rag_engine.mapping
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
 