"""    
This module implements the core RAG (Retrieval-Augmented Generation) engine for the AI compliance auditing system. It provides functionality to initialize the RAG components, evaluate individual requirements against a given document, and perform a full audit of a document by iterating through all mapped requirements. The engine uses a combination of a vector database (Qdrant) for retrieving relevant regulatory context, and a language model (e.g. Langchain's ChatOpenAI) to perform the actual evaluation based on a structured prompt template. The results are returned in a structured format that includes scores and auditor notes for each requirement.

"""

import os
import json
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# New modular imports
from backend.core.retrieval import RetrievalEngine
from backend.core.evaluation import EvaluationEngine

load_dotenv()

PARAMS_PATH = os.environ.get("PARAMS_PATH")
if not PARAMS_PATH: # If PARAMS_PATH is not set, assume we're running in a container with a fixed path, otherwise use the provided path (e.g., for local development)
    PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # container root is /app or local root where this file is located
    PARAMS_PATH = os.path.join(PROJECT_ROOT, "params.yaml")
else: # If PARAMS_PATH is set, we assume it's an absolute path provided via env var (e.g., for local development)
    PROJECT_ROOT = os.path.abspath(os.path.dirname(PARAMS_PATH))

def load_params(path: Optional[str] = None) -> Dict[str, Any]:
    params_path = path or PARAMS_PATH
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


params = load_params()
vect_params = params.get("vectorization", {})
rag_params = params.get("rag", {})



RequirementScore = Union[int, str] # Score can be an integer from 0 to 5, or "N/A" if not applicable or if parsing fails

class SubRequirementReport(BaseModel):
    Reference: str
    Source: str
    Score: RequirementScore
    Rationale: str
    Auditor_Notes: str
    Contexts: List[str]

class RequirementReport(BaseModel):
    Requirement_ID: str
    Requirement_Category: str
    Requirement_Name: str
    Score: RequirementScore
    Rationale: Optional[str] = None
    Auditor_Notes: str
    Prompt: str  # aggregation prompt used for this requirement
    Context: Optional[List[str]] = None # We can include the providex doc context for debug steps
    SubRequirements: List[SubRequirementReport] = []


class AuditResponse(BaseModel):
    requirements: List[RequirementReport] #We can include the providex doc context for future steps



vector_db: Optional[QdrantClient] = None
mapping: Optional[Dict[str, Any]] = None
llm: Optional[ChatOpenAI] = None
embedding_model: Optional[SentenceTransformer] = None
requirement_chunks: Dict[str, Any] = {}

# New engines
retrieval_engine: Optional[RetrievalEngine] = None
evaluation_engine: Optional[EvaluationEngine] = None

_initialized = False # Flag to prevent re-initialization if already done


def _candidate_paths(relative_path: str) -> List[str]: # Generate candidate paths for a given relative path, including both relative and absolute forms, and ensure uniqueness while preserving order
    rel = relative_path.replace("\\", "/")
    candidates = [
        os.path.join(PROJECT_ROOT, rel),
        os.path.abspath(os.path.join(PROJECT_ROOT, rel)),
        os.path.abspath(relative_path),
    ]
    unique: List[str] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return unique


# Initialization function to set up vector DB, mapping, requirement chunks, and LLM. It checks for environment variables to determine how to connect to Qdrant (external service vs embedded index) and loads necessary data files. The function can be forced to re-initialize if needed.
def init_rag(force: bool = False) -> None:
    global vector_db, llm, embedding_model, requirement_chunks, _initialized, retrieval_engine, evaluation_engine
    if _initialized and not force:
        return

    try:

        # If environment variables for Qdrant connection are set, use them to connect to an external Qdrant service. Otherwise, look for a local embedded index file. This allows flexibility for different deployment scenarios (local development vs production).
        qdrant_host = os.environ.get("QDRANT_HOST")
        qdrant_port = os.environ.get("QDRANT_PORT")
        if qdrant_host or qdrant_port:
            host = qdrant_host or "qdrant"
            port = int(qdrant_port) if qdrant_port else 6333
            try:
                vector_db = QdrantClient(host=host, port=port)
                print(f"✓ RAG Initialized with Qdrant service at {host}:{port}")
            except Exception as e:
                print(f"⚠ Qdrant service not available at {host}:{port}: {e}")
                vector_db = None
        else:
            # Embedded/local mode: use vector_index as a file
            index_path = vect_params.get("vector_index_path", "data/processed/vector_index")
            final_index_path = next((p for p in _candidate_paths(index_path) if os.path.exists(p)), None)
            if final_index_path:
                vector_db = QdrantClient(path=final_index_path)
                print(f"✓ RAG Initialized with embedded Qdrant at {final_index_path}")
            else:
                print("⚠ Vector index not found! Proceeding without vector DB.")
                vector_db = None



        chunks_candidates = [
            os.path.join(PROJECT_ROOT, "data", "processed", "requirement_chunks.json"),
            os.path.join("data", "processed", "requirement_chunks.json"),
        ]
        chunks_path = next((p for p in chunks_candidates if os.path.exists(p)), None)
        if chunks_path:
            with open(chunks_path, "r", encoding="utf-8") as f:
                requirement_chunks = json.load(f)
            print(f"✓ Requirement chunks loaded from {chunks_path}")
        else:
            print("⚠ Requirement chunks file not found!")
            requirement_chunks = {}


        llm_model_name = rag_params.get("llm_model")
        llm_temperature = float(rag_params.get("llm_temperature", 0))
        llm = ChatOpenAI(model=llm_model_name, temperature=llm_temperature, request_timeout=30)
        print(f"✓ LLM Initialized ({llm_model_name}, temp={llm_temperature})")

        # Initialize embedding model ONCE
        embedding_model_name = vect_params.get("model_name")
        embedding_model = SentenceTransformer(embedding_model_name)
        print(f"✓ Embedding model initialized: {embedding_model_name}")

        # Initialize engines
        evaluation_engine = EvaluationEngine(llm)
        # Note: RetrievalEngine needs a doc_client, but that's per-document (in-memory).
        # We can instantiate it dynamically or pass None and set it later.
        # But wait, RetrievalEngine handles queries. It needs the embedding model.
        # The doc_client is passed to the query method, so we can init RetrievalEngine here with just the model if we redesign it slightly,
        # OR we just use it as a helper class instantiated per request?
        # Let's keep it consistent: Init one "Service" or helper.
        # But wait, the original `_query_qdrant` took `doc_client` as arg.
        # So `RetrievalEngine` methods should probably take `doc_client`.
        # Let's adjust RetrievalEngine to be a stateless service or initialized with global dependencies.

        # Actually, RetrievalEngine was designed in step 1 to take `doc_client` in __init__.
        # That means we need to instantiate it *per audit* or *per evaluation*.
        # Let's instantiate a global helper if possible, or just keep the class definition available.
        # Ideally, `RetrievalEngine` holds the embedding model.
        retrieval_engine = RetrievalEngine(doc_client=None, embedding_model=embedding_model)
        # We will override doc_client in the method call or set it on the instance before use.
        # Better design: pass doc_client to `query_for_requirement`. Let's assume I did that in step 1 (I did).
        # Wait, I checked step 1 code:
        # `class RetrievalEngine: def __init__(self, doc_client: QdrantClient, embedding_model: Optional[SentenceTransformer] = None): ...`
        # `def query_for_requirement(self, collection_name: str, ...)` -> it uses `self.doc_client`.
        # So I must instantiate it with the doc_client.
        # Since doc_client is created in `audit_document`, I should instantiate RetrievalEngine there.

        _initialized = True
    except Exception as exc:
        print(f"⚠ Init Error: {exc}")
        _initialized = False
        raise


def rag_ready() -> bool:
    return all([
        vector_db is not None,
        llm is not None,
        requirement_chunks is not None,
        embedding_model is not None
    ])


def _get_requirement_chunks_from_qdrant(requirement_name: str, regulatory_client: QdrantClient, regulatory_collection: str) -> list:
    """
    Retrieve all regulatory chunks (already embedded) for a given requirement from Qdrant.
    Args:
        requirement_name: The name of the requirement.
        regulatory_client: QdrantClient instance for the regulatory chunks DB.
        regulatory_collection: Name of the Qdrant collection with regulatory chunks.
    Returns:
        List of dicts with 'content', 'reference', etc.
    """
    from qdrant_client.http import models as qmodels
    hits = regulatory_client.scroll(
        collection_name=regulatory_collection,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="requirementName",
                match=qmodels.MatchValue(value=requirement_name)
            )]
        ),
        with_vectors=True,
        limit=100
    )[0]
    # Return both payload and embedding for each chunk
    return [dict(h.payload, embedding=h.vector) for h in hits]


def evaluate_requirement(
    requirement_data: Dict[str, Any],
    _doc_client: Any = None,
    _temp_collection: str = None,
    _regulatory_client: Any = None,
    _regulatory_collection: str = None,
) -> RequirementReport:
    """
    Evaluate a single requirement against the provided document text.

    """
    if not rag_ready():
        raise RuntimeError("RAG system not initialized")

    # 1. Retrieve already embedded regulatory chunks for the requirement from Qdrant regulatory collection.
    if _regulatory_client and _regulatory_collection:
        requirement_name = requirement_data.get("requirementName", "unknown")
        requirement_ethical_principle = requirement_data.get("ethicalPrinciple", "unknown")
        req_chunks_embeddings = _get_requirement_chunks_from_qdrant(requirement_name, _regulatory_client, _regulatory_collection)
        # Giving the particular requirement we obtain a list of regolatory chunks (with their embeddings) that are relevant for that requirement.

        print(f"Retrieved {len(req_chunks_embeddings)} regulatory chunks for requirement '{requirement_name}' ({requirement_ethical_principle}) from Qdrant collection '{_regulatory_collection}'.")
    else:
        # Manage failure, return error 
        print("⚠ Regulatory client or collection not provided, cannot retrieve requirement chunks from Qdrant. ")
        return RequirementReport(
            Requirement_ID=requirement_data.get("id", ""),
            Requirement_Category=requirement_data.get("ethicalPrinciple", "unknown"),
            Requirement_Name=requirement_name,
            Score=0,
            Auditor_Notes="Failed to retrieve regulatory chunks from Qdrant.",
            Prompt="",
            Context=None,
            SubRequirements=[]
        )

    # 2. Select relevant document chunks using RetrievalEngine
    # Instantiate RetrievalEngine for this document context
    # Note: We use the global embedding_model
    retriever = RetrievalEngine(doc_client=_doc_client, embedding_model=embedding_model)
    
    pre_rerank_top_k = int(rag_params.get("pre_rerank_top_k", 10))
    
    top_doc_chunks_by_group = retriever.query_for_requirement(
        collection_name=_temp_collection,
        req_chunks_embeddings=req_chunks_embeddings,
        top_k=pre_rerank_top_k
    )

    # Optimization: deduplication of retrieved chunks across groups
    unique_chunks = {}  # chunk_id -> {"content": str, "refs": set}
    for group, chunks in top_doc_chunks_by_group.items():
        for c in chunks:
            cid = c['chunk_id']
            if cid not in unique_chunks:
                unique_chunks[cid] = {
                    "content": c['content'],
                    "refs": {group}
                }
            else:
                unique_chunks[cid]["refs"].add(group)

    # Build context (optional, for debug or legacy reasons)
    doc_context_list = []
    for cid, data in unique_chunks.items():
        doc_context_list.append(data['content'])


    # 3. Evaluate Sub-requirements using EvaluationEngine
    sub_results = []
    sub_reports = []

    # Instantiate EvaluationEngine (global llm)
    evaluator = EvaluationEngine(llm)

    for reg_chunk in req_chunks_embeddings:
        reference = reg_chunk.get("reference", "")
        content = reg_chunk.get("content", "")
        # Fallback: if content is missing, try control + implementation_guidance, and always label them if present
        control = reg_chunk.get("control", "")
        guidance = reg_chunk.get("implementation_guidance", "")
        regulatory_parts = []
        if content:
            regulatory_parts.append(content)
        if control:
            regulatory_parts.append(f"[CONTROL] {control}")
        if guidance:
            regulatory_parts.append(f"[IMPLEMENTATION_GUIDANCE] {guidance}")
        content = "\n".join(regulatory_parts).strip()

        # Find document chunks associated with this reference
        relevant_chunks = []
        for cid, data in unique_chunks.items():
            if reference in data["refs"]:
                relevant_chunks.append(data["content"])

        sub_req_data = {
            "name": reference,
            "regulatory_content": content
        }

        # Use EvaluationEngine to evaluate
        # Note: evaluate_sub_requirement in EvaluationEngine takes (sub_req_name, regulatory_reference, associated_chunks)
        # wait, my EvaluationEngine.evaluate_sub_requirement signature in previous step:
        # def evaluate_sub_requirement(self, sub_req_name: str, regulatory_reference: str, associated_chunks: List[str]) -> Dict[str, Any]:

        result = evaluator.evaluate_sub_requirement(
            sub_req_name=reference,
            regulatory_reference=content,
            associated_chunks=relevant_chunks
        )

        # Compose context for RAGAS (legacy / tracking)
        ragas_contexts = []
        if content:
            reg_name = reference
            reg_source = reg_chunk.get("source", "")
            ragas_contexts.append(f"[REGULATORY] [NAME: {reg_name}] [SOURCE: {reg_source}] {content}")

        # Use top_doc_chunks_by_group to get relevant document chunks for this group
        group_key = None
        # Try both AI_ACT and ISO_42001 prefixes, fallback to reference only
        if reg_chunk.get('source') == 'EU_AI_ACT':
            group_key = f"AI_ACT::{reference}"
        elif reg_chunk.get('source') == 'ISO_42001':
            group_key = f"ISO_42001::{reference}"
        else:
            group_key = reference
        doc_chunks_for_group = top_doc_chunks_by_group.get(group_key, [])
        if doc_chunks_for_group:
            for chunk in doc_chunks_for_group:
                ragas_contexts.append(f"[DOCUMENT] {chunk['content']}")
        else:
            # Fallback: include all document chunks
            for cid, data in unique_chunks.items():
                ragas_contexts.append(f"[DOCUMENT] {data['content']}")

        # We need to reconstruct the result dict expected by aggregate_results
        # Combine rationale and notes for RAGAS evaluation to improve groundedness
        combined_answer = f"{result.get('rationale', '')}\n\nSummary: {result.get('auditor_notes', '')}"

        sub_results.append({
            "reference": reference,
            "source": reg_chunk.get("source", ""),
            "prompt": evaluator._get_sub_prompt(reference, content, relevant_chunks), # Re-generating prompt just for logging? Ideally EvaluationEngine returns it.
            "ragas_question": result.get("ragas_question", ""),
            "answer": combined_answer,
            "contexts": ragas_contexts,
            "score": result.get("score", "N/A"),
            "rationale": result.get("rationale", "")
        })

        sub_reports.append(SubRequirementReport(
            Reference=reference,
            Source=reg_chunk.get("source", ""),
            Score=result.get("score", "N/A"),
            Rationale=result.get("rationale", ""),
            Auditor_Notes=result.get("auditor_notes", ""),
            Contexts=ragas_contexts
        ))

    # 4. Aggregate results
    agg_result = evaluator.aggregate_results(sub_results)

    # Convert sub_results (list of dicts) to list of JSON strings for Pydantic validation (legacy field)
    context_strings = [json.dumps(sr, ensure_ascii=False) for sr in sub_results]

    return RequirementReport(
        Requirement_ID=requirement_data.get("id", ""),
        Requirement_Category=requirement_data.get("ethicalPrinciple", "unknown"),
        Requirement_Name=requirement_name,
        Score=agg_result.get("score", "N/A"),
        Auditor_Notes=agg_result.get("auditor_notes", ""),
        Rationale=agg_result.get("rationale", ""),
        Prompt=agg_result.get("prompt", ""),
        Context=context_strings,
        SubRequirements=sub_reports
    )


# Main function to perform a full audit of a document by evaluating all requirements in the mapping
def audit_document(
    document_text: str,
    *,
    debug_dump_path: Optional[str] = None,
    requirement_limit: Optional[int] = None,
) -> AuditResponse:
    if not rag_ready():
        raise RuntimeError("RAG system not initialized")
    assert requirement_chunks is not None

    # 1. Chunk and embed the document under test ONCE
    if embedding_model is None:
        raise RuntimeError("Embedding model not initialized")
    
    document_chunk_size = rag_params.get("document_chunk_size", 512)
    document_chunk_overlap = rag_params.get("document_chunk_overlap", 64)

    doc_chunks = _chunk_document(document_text, chunk_size=document_chunk_size, chunk_overlap=document_chunk_overlap)
    doc_embs = _embed_chunks(doc_chunks, embedding_model)


    #Store document chunks and embeddings in a temporary in-memory Qdrant collection for efficient retrieval during requirement evaluation. This avoids the need for file-based storage and cleanup issues, while still allowing us to leverage Qdrant's vector search capabilities.
    temp_collection = f"temp_doc_{os.getpid()}_audit"
    # Use in-memory Qdrant for doc_client to avoid file locking and cleanup issues
    doc_client = QdrantClient(":memory:")

    vector_size = embedding_model.get_sentence_embedding_dimension()
    doc_client.recreate_collection(
        collection_name=temp_collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    points = [PointStruct(id=i, vector=doc_embs[i].tolist(), payload={
        'content': doc_chunks[i], 'chunk_id': i
    }) for i in range(len(doc_chunks))]
    doc_client.upsert(collection_name=temp_collection, points=points)


    requirements_reports: List[RequirementReport] = []

    # Instantiate regulatory_client and regulatory_collection for regulatory chunks
    regulatory_collection = vect_params.get("regulatory_collection", "legal_docs")
    # Use the same vector_db if already initialized, otherwise create a new client (service or embedded)
    if vector_db is not None:
        regulatory_client = vector_db
    else:
        # Fallback: use embedded local index
        index_path = vect_params.get("vector_index_path", "data/processed/vector_index")
        regulatory_client = QdrantClient(path=index_path)

    # For each requirement in requirement_chunks (list), call evaluate_requirement to get the report for that requirement and collect all reports in a list.

    req_iter = requirement_chunks
    if requirement_limit is not None:
        req_iter = req_iter[:requirement_limit]

    for req_data in req_iter:
        requirement_report = evaluate_requirement(
            requirement_data=req_data,
            _doc_client=doc_client,
            _temp_collection=temp_collection,
            _regulatory_client=regulatory_client,
            _regulatory_collection=regulatory_collection
        )
        requirements_reports.append(requirement_report)

    if debug_dump_path: # If a debug dump path is provided, save the raw requirement reports
        debug_dir = os.path.dirname(debug_dump_path) or "."
        os.makedirs(debug_dir, exist_ok=True)
        with open(debug_dump_path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in requirements_reports], f, ensure_ascii=False, indent=4)


    # Cleanup in-memory Qdrant client
    doc_client.delete_collection(collection_name=temp_collection)
    if hasattr(doc_client, 'close'):
        doc_client.close()

    return AuditResponse(requirements=requirements_reports)


__all__ = [
    "AuditResponse",
    "RequirementReport",
    "audit_document",
    "evaluate_requirement",
    "init_rag",
    "load_params",
    "rag_ready",
]

# Helper functions
def _chunk_document(document_text: str, chunk_size: int = 512, chunk_overlap: int = 64):
    """
    Split the input document text into overlapping chunks for embedding.
    Args:
        document_text: The full text of the document to split.
        chunk_size: The size of each chunk (in characters).
        chunk_overlap: The overlap between consecutive chunks (in characters).
    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(document_text)
    return chunks


def _embed_chunks(chunks: List[str], model: SentenceTransformer):
    """
    Generate embeddings for a list of text chunks using the provided model.
    Args:
        chunks: List of text strings to embed.
        model: SentenceTransformer model instance.
    Returns:
        Numpy array of embeddings.
    """
    return model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


if __name__ == "__main__":
    init_rag()
    # check qdrant regulatory retrieval for a sample requirement
    points = _get_requirement_chunks_from_qdrant("transparency", vector_db, vect_params.get("regulatory_collection", "legal_docs"))

    print (f"Retrieved {len(points)} regulatory chunks for 'Transparency' requirement:")
    for p in points:
        print(p)