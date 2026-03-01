import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class EvaluationEngine:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _get_sub_prompt(self, main_req_name: str, reference: str, content: str, relevant_chunks: List[str]) -> str:
        return f"""
You are an expert AI auditor specializing in EU AI Act and ISO 42001 compliance.
Your task is to assess whether the provided document chunks demonstrate coverage of a specific sub-requirement, which is part of the broader requirement: '{main_req_name}'.

SUB-REQUIREMENT: {reference}
REGULATORY CONTEXT: {content}

DOCUMENT CHUNKS:
{chr(10).join(relevant_chunks)}

INSTRUCTIONS:
1. Analyze the chunks carefully. Look for ANY explicit mentions OR implicit evidence that addresses the regulatory context.
2. Be objective. If the chunks provide partial or related evidence, explain how it relates to the requirement rather than just saying "no evidence".
3. Support your reasoning by referencing specific parts of the text (e.g., "The document states that...").
4. Only score as 0 if the chunks are completely irrelevant to the regulatory context.

Respond in JSON format:
{{
    "rationale": "Detailed explanation of findings referencing the provided text.",
    "score": "Integer 0-5 (0=No evidence, 5=Fully compliant) or 'N/A'",
    "auditor_notes": "Concise summary for the final report."
}}
"""

    def _get_aggregate_prompt(self, sub_results: List[Dict[str, Any]]) -> str:
        simplified_results = []
        for res in sub_results:
            simplified_results.append({
                "reference": res.get("reference"),
                "score": res.get("score"),
                "auditor_notes": res.get("auditor_notes", res.get("answer", "")) # Use concise auditor_notes
            })

        return f"""
You are a Lead AI Auditor consolidating findings for a main requirement based on several sub-requirement evaluations.

SUB-REQUIREMENT FINDINGS:
{json.dumps(simplified_results, indent=2, ensure_ascii=False)}

TASK:
Aggregate these findings into a final assessment for the main requirement.

INSTRUCTIONS:
1. The overall score should reflect the weakest links. If critical sub-requirements are missing, the score should be low.
2. The 'auditor_notes' must be a structured summary listing the status of key sub-requirements (e.g., "Article 10: Covered; Article 13: Missing").
3. The 'rationale' should provide a high-level justification.

Respond in JSON format:
{{
    "score": "Integer 0-5 or 'N/A'",
    "auditor_notes": "Structured summary of coverage across sub-requirements.",
    "rationale": "High-level justification for the overall score."
}}
"""

    def evaluate_sub_requirement(self, main_req_name: str, sub_req_name: str, regulatory_reference: str, associated_chunks: List[str]) -> Dict[str, Any]:
        """
        Evaluates a single sub-requirement using the LLM.
        """
        prompt = self._get_sub_prompt(main_req_name, sub_req_name, regulatory_reference, associated_chunks)
        # Simplify the RAGAS question to avoid redundancy and improve semantic matching (Relevancy)
        ragas_question = f"Does the document provide evidence of compliance for the '{main_req_name}' sub-requirement '{sub_req_name}'?"
        response = self.llm.invoke(prompt).content.strip()
        try:
            cleaned = response.replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned)
        except Exception:
            result = {"score": "N/A", "rationale": "Parsing failed", "auditor_notes": response}

        result["ragas_question"] = ragas_question
        return result

    def aggregate_results(self, sub_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates multiple sub-requirement results into a final report.
        """
        agg_prompt = self._get_aggregate_prompt(sub_results)
        agg_response = self.llm.invoke(agg_prompt).content.strip()
        try:
            cleaned_agg = agg_response.replace("```json", "").replace("```", "").strip()
            agg_result = json.loads(cleaned_agg)
        except Exception:
            agg_result = {"score": "N/A", "auditor_notes": "Parsing failed", "rationale": agg_response}

        # Helper: ensure auditor_notes is a string
        notes_val = agg_result.get("auditor_notes", "")
        if isinstance(notes_val, (dict, list)):
            try:
                # If it's a dict/list, dump it as formatted JSON text
                notes_str = json.dumps(notes_val, indent=2, ensure_ascii=False)
            except:
                notes_str = str(notes_val)
        else:
            notes_str = str(notes_val)

        return {
            "score": agg_result.get("score", "N/A"),
            "auditor_notes": notes_str,
            "rationale": agg_result.get("rationale", ""),
            "prompt": agg_prompt
        }
