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


# Static prompt template 
PROMPT_TEMPLATE = """
You are a Senior AI Compliance Auditor. Your task is to perform a strict, evidence-based gap analysis using a "Guidance-based Assessment" logic.

DOCUMENT TO EVALUATE (chunks from the original document, retrieved based on relevance to the requirement being evaluated, and tagged with [[Next chunk is relevant for: ...]] to indicate which regulatory requirement(s) they pertain to):
{document_text}

REQUIREMENT SUMMARY:
Requirement name: {requirement_name}

This requirement is associated with the following regulatory references (AI Act articles and ISO 42001 sections). Use these as the basis for your evaluation.
Requirement references: {requirement_references}

REGULATORY CONTEXT (The ONLY source of truth for your evaluation. Use this to understand the requirement and to identify what evidence in the document would correspond to different levels of compliance):
{regulatory_references}

AUDIT RULES (MANDATORY):
1. STRICT ANCHORING (Faithfulness): The compliance score and verdict MUST be derived EXCLUSIVELY from the provided 'REGULATORY CONTEXT'. Do NOT use external knowledge, prior training, or general AI Act/ISO assumptions. If a requirement is not in the provided text, do not evaluate it.
2. NO HALLUCINATION: Do not deduce, infer, or assume missing procedures. If an objective, metric, or procedure is not explicitly written in the DOCUMENT TO EVALUATE, it does not exist.
3. EXPLICIT MAPPING (Groundedness): You must explicitly quote the REGULATORY CONTEXT and map it directly to the tagged evidence ([[Next chunk is relevant for: ...]]) in the DOCUMENT TO EVALUATE.

SCORING CRITERIA:
5 (Full Compliance): Perfect alignment with the provided regulatory context; specific technical evidence is present.
4 (Substantial Compliance): Solid implementation, but lacks secondary details requested by the regulation.
3 (Partial Compliance): Some evidence present, but significant procedural gaps exist against the regulation.
2 (Major Gap): Minimal evidence; major aspects of the regulation are ignored.
1 (Negligible): The requirement is mentioned, but without substantive evidence.
0 (No Compliance): The document is completely silent on the requirement.

RESPONSE FORMAT (JSON ONLY):
{{
    "rationale": "For each claim about compliance or non-compliance, use the following format, separating each claim with two newlines. For each claim: \n\nClaim: <state the claim concisely>\nSupported by: <quote or cite specific document chunks using the provided tags, e.g. [[Next chunk is relevant for: ...]] ...chunk text...; if not supported, write 'None'>\nRegulatory reference: <quote the exact requirement/guidance from the Regulatory Context>\n\nRepeat for each claim. If a claim is not supported by any chunk, 'Supported by' must be 'None'.",
    "score": "integer (0-5) or 'N/A'",
    "auditor_notes": "Start EXACTLY with 'The document is [compliant / partially compliant / not compliant] with {requirement_name}.' Then provide a comprehensive summary that explicitly names the regulatory reference driving your conclusion (e.g., 'As required by ISO 42001 B.3.2...') and explicitly states what specific evidence was found or is missing in the evaluated document. Ensure the language is natural and conversational. Max 120 words."
}}
"""

RequirementScore = Union[int, str] # Score can be an integer from 0 to 5, or "N/A" if not applicable or if parsing fails

class RequirementReport(BaseModel):
    Requirement_ID: str
    Requirement_Category: str
    Requirement_Name: str
    Score: RequirementScore
    Rationale: Optional[str] = None
    Auditor_Notes: str
    Prompt: str  # static prompt included for monitoring/debugging purposes
    Context: Optional[List[str]] = None # We can include the providex doc context for debug steps
    


class AuditResponse(BaseModel):
    requirements: List[RequirementReport] #We can include the providex doc context for future steps



vector_db: Optional[QdrantClient] = None
mapping: Optional[Dict[str, Any]] = None
llm: Optional[ChatOpenAI] = None
embedding_model: Optional[SentenceTransformer] = None
requirement_chunks: Dict[str, Any] = {}
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
    global vector_db, llm, embedding_model, requirement_chunks, _initialized 
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
    ])


def _build_requirement_text(req_data: Dict[str, Any]) -> str:
    """
    Build a requirement text for prompting, using the new structure from requirement_chunks.json.
    - requirementName: the main requirement name/title
    - euAiActArticles: list of dicts with 'reference' and 'content' (AI Act articles)
    - iso42001Reference: list of dicts with 'reference' and 'content' (ISO references)
    """
    req_name = req_data.get("requirementName", "")
    ai_act_text = "\n".join([
        f"[AI Act: {art.get('reference', '')}]" for art in req_data.get("euAiActArticles", [])
    ])
    iso_text = "\n".join([
        f"[ISO 42001: {iso.get('reference', '')}] " for iso in req_data.get("iso42001Reference", [])
    ])
    # Compose all together for the prompt
    return f"Requirement: {req_name}\n\nAI Act References:\n{ai_act_text}\n\nISO 42001 References:\n{iso_text}".strip()


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
            Prompt=PROMPT_TEMPLATE.strip(),
            Context=None
        )

    # 2. Select relevant document chunks by querying Qdrant with the requirement chunks as queries. This retrieves the most relevant parts of the document that pertain to the requirement being evaluated.

    doc_client = _doc_client # The Qdrant client 
    temp_collection = _temp_collection
    
    # Read the pre-rerank top-k parameter from params.yaml
    pre_rerank_top_k = int(rag_params.get("pre_rerank_top_k", 10)) 

    
    # We pass the list of regulatory chunks (with their embeddings) to query Qdrant and retrieve the most relevant document chunks for that requirement. The retrieved chunks are grouped by the regulatory chunk they are relevant to, which allows us to maintain the mapping between the regulatory context and the evidence in the document.
    top_doc_chunks_by_group = _query_qdrant_for_requirement(doc_client, temp_collection, req_chunks_embeddings, top_k=pre_rerank_top_k)

    # e.g Data Governance requirement has 3 regulatory chunks (e.g. 3 AI Act articles), for each of them we query Qdrant and we obtain a list of the most relevant document chunks for each regulatory chunk, so we have a mapping like this:
    # {
    #     "AI Act Article 1": [chunk1, chunk2, ...],
    #     "AI Act Article 2": [chunk3, chunk4, ...],
    #     "ISO 42001 Section 5.1": [chunk5, chunk6, ...],
    # }

    # Now, in order to reduce redunancy in the prompt we organize the data structure for chunks, keeping track for each of the retrieved document chunk which regulatory chunk(s) it was relevant for. 

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

    # E.g if chunk1 was relevant for both "AI Act Article 1" and "ISO 42001 Section 5.1", we will have:
    # unique_chunks = {
    #     "chunk1_id": {
    #         "content": "text of chunk1",
    #         "refs": {"AI Act Article 1", "ISO 42001 Section 5.1"}
    #     },
    #     ...

    # Build context
    doc_context_list = []
    for cid, data in unique_chunks.items():
        refs_str = ", ".join(sorted(list(data['refs'])))
        chunk_entry = f"[[Next chunk is relevant for: {refs_str}]]\n{data['content']}"
        doc_context_list.append(chunk_entry)

    doc_context = "\n\n".join(doc_context_list)
    #all_doc_chunks = [data['content'] for data in unique_chunks.values()]

    # 3. re-ranking (skipped)
    # top_k = int(rag_params.get("top_k", 4))

    # 4. Build the prompt and call the LLM
    regulatory_context_list = [f"[{c.get('reference', '')}] {c.get('content', '')}" for c in req_chunks_embeddings]

    # Build a list that cointains the whole provided context for RAGAS purposes
    eval_context_list = doc_context_list + regulatory_context_list 
    
    regulatory_context = "\n".join([c.get('content', '') for c in req_chunks_embeddings])

    prompt = PROMPT_TEMPLATE.format(
        document_text=doc_context.strip(),
        requirement_name=requirement_name,
        requirement_references=_build_requirement_text(requirement_data),
        regulatory_references=regulatory_context,
    )

    # Debug for only one requirement to avoid too much logs, we can focus on the "Data Governance" requirement as an example, or if the retrieved context is empty (which is a critical case to debug)
    #if (requirement_name == "Risks" ):
    #    print(f"Evaluating requirement '{requirement_name}' with prompt:\n{prompt}\n") # Debug: print the #prompt being sent to the LLM

    assert llm is not None
    llm_response = llm.invoke(prompt).content.strip()
    try:
        cleaned_response = llm_response.replace("```json", "").replace("```", "").strip()
        response_json = json.loads(cleaned_response)
        score_0_5 = int(response_json["score"])
        auditor_notes = response_json["auditor_notes"]
        rationale = response_json.get("rationale", "")
    except Exception:
        score_0_5 = 0
        auditor_notes = f"LLM response parsing failed. Response was: {llm_response}"
        rationale = ""
    return RequirementReport(
        Requirement_ID=requirement_data.get("id", ""),
        Requirement_Category=requirement_data.get("ethicalPrinciple", "unknown"),
        Requirement_Name=requirement_name,
        Score=score_0_5 if isinstance(score_0_5, int) else "N/A",
        Auditor_Notes=auditor_notes,
        Rationale=rationale,
        Prompt=prompt,
        Context= eval_context_list, 
    )

# Main function to perform a full audit of a document by evaluating all requirements in the mapping
def audit_document(
    document_text: str,
    *,
    debug_dump_path: Optional[str] = None,
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
    for req_data in requirement_chunks:
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


def _query_qdrant_for_requirement(doc_client: QdrantClient, doc_collection: str, req_chunks_embeddings: List[dict], top_k: int = 4):
    
    from collections import defaultdict
    group_to_chunks = defaultdict(list)
    #print(req_chunks_embeddings[0].keys() if req_chunks_embeddings else "Empty list")
    for req in req_chunks_embeddings: # For each requirement
        reference = req.get("reference") # 
        source = req.get("source") 
        assigned = False
        if reference and source:
            if source == "EU_AI_ACT":
                group_to_chunks[f"AI_ACT::{reference}"].append(req)
                assigned = True
            elif source == "ISO_42001":
                group_to_chunks[f"ISO_42001::{reference}"].append(req)
                assigned = True
        if not assigned:
            group_to_chunks['NO_GROUP'].append(req)

    results_by_group = {}
    for group, chunks in group_to_chunks.items():
        group_results = []
        for req in chunks:
            req_emb = req['embedding']
            hits = doc_client.query_points(collection_name=doc_collection, query=req_emb, limit=top_k).points
            for hit in hits:
                group_results.append({
                    'score': hit.score,
                    'content': hit.payload.get('content', ''),
                    'chunk_id': hit.payload.get('chunk_id', None)
                })
        # Deduplicate by chunk_id, keep only the highest score
        seen = {}
        for r in group_results:
            cid = r['chunk_id']
            if cid not in seen or r['score'] > seen[cid]['score']:
                seen[cid] = r
        # Take top_k for each group
        top_chunks = list(seen.values())
        top_chunks.sort(key=lambda x: x['score'], reverse=True)
        results_by_group[group] = top_chunks[:top_k]

    return results_by_group


if __name__ == "__main__":
    init_rag()
    # check qdrant regulatory retrieval for a sample requirement
    points = _get_requirement_chunks_from_qdrant("transparency", vector_db, vect_params.get("regulatory_collection", "legal_docs"))

    print (f"Retrieved {len(points)} regulatory chunks for 'Transparency' requirement:")
    for p in points:
        print(p)