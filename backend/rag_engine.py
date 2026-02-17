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
llm_params = params.get("llm", {})


# Static prompt template 
PROMPT_TEMPLATE = """
You are a Senior AI Compliance Auditor. Your task is to perform a professional gap analysis using a "Guidance-based Assessment" logic.

DOCUMENT TO EVALUATE:
{document_text}

REQUIREMENT CATEGORY:
{requirement_text}

REGULATORY CONTEXT:
{regulatory_references}

AUDIT RULES (MANDATORY):
1. ANCHORING: The compliance score and verdict MUST be based ONLY on 'AI ACT' Articles and 'ISO Annex A' (ID: A.x.x). You must NEVER declare non-compliance against Annex B, as it is not mandatory.
2. GUIDANCE DEPENDENCY: Use 'ISO Annex B' (ID: B.x.x) as the lens to evaluate the mandatory 'ISO Annex A' requirement. 
   - Example: To judge if ISO A.3.2 (Roles) is met, use the details in ISO B.3.2 to verify if the document provides sufficient implementation evidence.
3. EVIDENCE RECOGNITION: Recognize specific metrics (e.g., %, F1-score), named tools, or documented procedures as high-level evidence.

SCORING CRITERIA:
5 (Full Compliance): Meets the AI Act/ISO Annex A requirement perfectly, providing the specific technical evidence (metrics/tools) suggested in the corresponding ISO Annex B.
4 (Substantial Compliance): Meets the mandatory AI Act/ISO Annex A requirement. The implementation is solid, though it may lack some secondary details described in Annex B.
3 (Partial Compliance): The mandatory Annex A requirement is addressed, but the implementation lacks the procedural depth or metrics recommended in Annex B to be truly effective.
2 (Major Gap): The mandatory Annex A requirement is mentioned, but the document lacks the technical substance or the "how-to" described in Annex B.
1 (Negligible): Mentioned only in passing without any alignment with Annex A.
0 (No Compliance): The document is completely silent on the requirement.

OUTPUT INSTRUCTIONS:
- Start 'auditor_notes' with: "Compliance/Non-compliance with [AI Act Art. / ISO Annex A ID]". 
- DO NOT cite ISO B as the violated requirement. Instead, state: "Non-compliant with ISO A.x.x because [Evidence from B.x.x] is missing."
- Respond ONLY in valid JSON format.

RESPONSE FORMAT:
{{
    "score": integer (0-5),
    "auditor_notes": "Verdict vs Annex A/AI Act. Cite ID. Only for the Annex A requirement explain why based on Annex B guidance, for the AI Act use the corresponding article. Max 100 words."
}}
"""

RequirementScore = Union[int, str] # Score can be an integer from 0 to 5, or "N/A" if not applicable or if parsing fails

class RequirementReport(BaseModel):
    Mapped_ID: str
    Requirement_Name: str
    Score: RequirementScore
    Auditor_Notes: str
    Prompt: str  # static prompt included for monitoring/debugging purposes


class AuditResponse(BaseModel):
    requirements: List[RequirementReport]


vector_db: Optional[QdrantClient] = None
mapping: Optional[Dict[str, Any]] = None
llm: Optional[ChatOpenAI] = None
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
    global vector_db, mapping, llm, requirement_chunks, _initialized 
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

        mapping_path = os.path.join(PROJECT_ROOT, "data", "mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            print(f"✓ Mapping loaded with {len(mapping) if mapping else 0} requirements")
        else:
            print("⚠ Mapping file not found!")
            mapping = {}

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

        llm_model_name = llm_params.get("llm_model")
        llm_temperature = float(llm_params.get("llm_temperature", 0))
        llm = ChatOpenAI(model=llm_model_name, temperature=llm_temperature, request_timeout=30)
        print(f"✓ LLM Initialized ({llm_model_name}, temp={llm_temperature})")

        _initialized = True
    except Exception as exc:
        print(f"⚠ Init Error: {exc}")
        _initialized = False
        raise


def rag_ready() -> bool:
    return all([
        vector_db is not None,
        mapping is not None,
        llm is not None,
        bool(requirement_chunks),
    ])


def _build_requirement_text(req_data: Dict[str, Any]) -> str:
    iso_text = req_data.get("iso_control_text", "")
    ai_act_text = " ".join([art.get("text", "") for art in req_data.get("ai_act_articles", [])])
    return f"{iso_text} {ai_act_text}".strip()


def evaluate_requirement(
    document_text: str,
    requirement_name: str,
    requirement_data: Dict[str, Any],
) -> RequirementReport:
    if not rag_ready():
        raise RuntimeError("RAG system not initialized")

    req_text = _build_requirement_text(requirement_data) # Build the requirement text for the prompt
    pre_chunks = requirement_chunks.get(requirement_name, []) # Get the precomputed relevant chunks for this requirement from the loaded requirement_chunks data
    chunks_text = [chunk.get("content", "") for chunk in pre_chunks] # Extract the text content of the chunks to include in the prompt as regulatory references
    chunks_joined = "\n".join(chunks_text) # Join the chunk texts with newlines to create a single string to insert into the prompt under "REGULATORY CONTEXT". This provides the LLM with the relevant context for evaluating the requirement against the document.


    prompt = PROMPT_TEMPLATE.format( 
        document_text=document_text,
        requirement_text=req_text,
        regulatory_references=chunks_joined,
    )

    assert llm is not None
    llm_response = llm.invoke(prompt).content.strip() # Call the LLM with the constructed prompt and get the response. We expect the response to be a JSON string containing the score and auditor notes as per the instructions in the prompt.

    try:
        cleaned_response = llm_response.replace("```json", "").replace("```", "").strip() # Clean response
        response_json = json.loads(cleaned_response) # Parse the response as JSON
        score_0_5 = int(response_json["score"]) # Extract the score
        auditor_notes = response_json["auditor_notes"] # Extract the auditor notes
    except Exception:
        score_0_5 = 0
        auditor_notes = f"LLM response parsing failed. Response was: {llm_response}"

    return RequirementReport(
        Mapped_ID=requirement_data.get("id", "unknown"),
        Requirement_Name=requirement_name,
        Score=score_0_5 if isinstance(score_0_5, int) else "N/A",
        Auditor_Notes=auditor_notes,
        Prompt=PROMPT_TEMPLATE.strip(), # Include the static prompt template in the report for monitoring/debugging purposes, so we can see exactly what was sent to the LLM for each requirement evaluation.
    )

# Main function to perform a full audit of a document by evaluating all requirements in the mapping
def audit_document(
    document_text: str,
    *,
    debug_dump_path: Optional[str] = None,
) -> AuditResponse:
    if not rag_ready():
        raise RuntimeError("RAG system not initialized")
    assert mapping is not None

    requirements_reports: List[RequirementReport] = []

    # For each requirement in the mapping, call evaluate_requirement to get the report for that requirement and collect all reports in a list. This will produce a comprehensive audit report covering all requirements.
    for req_name, req_data in mapping.items():
        requirement_report = evaluate_requirement(document_text, req_name, req_data)
        requirements_reports.append(requirement_report)

    if debug_dump_path: # If a debug dump path is provided, save the raw requirement reports
        debug_dir = os.path.dirname(debug_dump_path) or "."
        os.makedirs(debug_dir, exist_ok=True)
        with open(debug_dump_path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in requirements_reports], f, ensure_ascii=False, indent=4)

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
