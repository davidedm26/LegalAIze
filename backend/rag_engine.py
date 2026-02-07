import os
import json
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARAMS_PATH = os.path.join(PROJECT_ROOT, "params.yaml")


def load_params(path: Optional[str] = None) -> Dict[str, Any]:
    params_path = path or PARAMS_PATH
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


params = load_params()
vect_params = params.get("vectorization", {})
eval_params = params.get("evaluation", {})

RequirementScore = Union[int, str]


class RequirementReport(BaseModel):
    Mapped_ID: str
    Requirement_Name: str
    Score: RequirementScore
    Auditor_Notes: str


class AuditResponse(BaseModel):
    requirements: List[RequirementReport]


embedding_model: Optional[SentenceTransformer] = None
vector_db: Optional[QdrantClient] = None
mapping: Optional[Dict[str, Any]] = None
llm: Optional[ChatOpenAI] = None
requirement_chunks: Dict[str, Any] = {}
_initialized = False


def _candidate_paths(relative_path: str) -> List[str]:
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


def init_rag(force: bool = False) -> None:
    global embedding_model, vector_db, mapping, llm, requirement_chunks, _initialized
    if _initialized and not force:
        return

    try:
        model_name = vect_params.get("model_name", "all-MiniLM-L6-v2")
        embedding_model = SentenceTransformer(model_name)

        index_path = vect_params.get("vector_index_path", "data/processed/vector_index")
        final_index_path = next((p for p in _candidate_paths(index_path) if os.path.exists(p)), None)
        if final_index_path:
            vector_db = QdrantClient(path=final_index_path)
            print(f"✓ RAG Initialized with index: {final_index_path}")
        else:
            print("⚠ Vector index not found! Proceeding without vector DB.")

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

        llm_model_name = eval_params.get("llm_model")
        llm_temperature = float(eval_params.get("llm_temperature", 0))
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
        embedding_model is not None,
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

    req_text = _build_requirement_text(requirement_data)
    pre_chunks = requirement_chunks.get(requirement_name, [])
    chunks_text = [chunk.get("content", "") for chunk in pre_chunks]
    chunks_joined = "\n".join(chunks_text)

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

    assert llm is not None
    llm_response = llm.invoke(prompt).content.strip()

    try:
        cleaned_response = llm_response.replace("```json", "").replace("```", "").strip()
        response_json = json.loads(cleaned_response)
        score_0_5 = int(response_json["score"])
        auditor_notes = response_json["auditor_notes"]
    except Exception:
        score_0_5 = 0
        auditor_notes = f"LLM response parsing failed. Response was: {llm_response}"

    return RequirementReport(
        Mapped_ID=requirement_data.get("id", "unknown"),
        Requirement_Name=requirement_name,
        Score=score_0_5 if isinstance(score_0_5, int) else "N/A",
        Auditor_Notes=auditor_notes,
    )


def audit_document(
    document_text: str,
    *,
    debug_dump_path: Optional[str] = None,
) -> AuditResponse:
    if not rag_ready():
        raise RuntimeError("RAG system not initialized")
    assert mapping is not None

    requirements_reports: List[RequirementReport] = []

    for req_name, req_data in mapping.items():
        requirement_report = evaluate_requirement(document_text, req_name, req_data)
        requirements_reports.append(requirement_report)

    if debug_dump_path:
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
