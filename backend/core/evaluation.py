import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class EvaluationEngine:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _get_sub_prompt(self, main_req_name: str, reference: str, source: str, content: str, relevant_chunks: List[str]) -> str:
        return f"""
You are an expert AI auditor specializing in EU AI Act and ISO 42001 compliance.
Your task is to assess whether the provided document chunks demonstrate coverage of a specific sub-requirement, which is part of the broader requirement: '{main_req_name}'.

SUB-REQUIREMENT: {reference} ({source})
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
                "source": res.get("source"),
                "score": res.get("score"),
                "auditor_notes": res.get("auditor_notes", res.get("answer", "")) # Use concise auditor_notes
            })

        return f"""
You are a Lead AI Auditor consolidating findings for a main requirement based on several sub-requirement evaluations.

SUB-REQUIREMENT FINDINGS:
{json.dumps(simplified_results, indent=2, ensure_ascii=False)}

TASK:
Aggregate these findings into a final compliance assessment for the main requirement.

INSTRUCTIONS:
1. Calculate an overall score reflecting the compliance level. If critical sub-requirements score low, the overall score should be proportionally low.

2. Write 'auditor_notes' as an EXECUTIVE SUMMARY for management/stakeholders (NO technical legal references):
   - Write in paragraph form (NOT a list, NO bullet points, NO semicolons separating items)
   - Start with overall compliance status (e.g., "Partially compliant", "Non-compliant", "Fully compliant")
   - Describe FUNCTIONALLY what was found and what gaps exist
   - Use business-friendly language: "accuracy measurements", "risk management processes", "oversight mechanisms"
   - DO NOT mention specific article numbers, paragraph numbers, or section codes
   - Focus on impact and actionable insights

3. Write 'rationale' as a TECHNICAL ANALYSIS for compliance experts (WITH legal references):
   - Reference specific sub-requirement codes and scores (e.g., "Article 15 Para 1 scored 2", "ISO 42001 section 6.1.1 scored 1")
   - Map findings to regulatory requirements precisely
   - Explain technical compliance implications
   - Be definitive: instead of "X is missing", say "The documentation does not provide X"

TONE: 
- auditor_notes: Executive, accessible, action-oriented
- rationale: Technical, precise, regulation-focused

EXAMPLE FORMAT for auditor_notes (NO legal references):
"The system demonstrates partial compliance with robustness and safety requirements. Documentation shows awareness of the AI's high-risk classification and includes some automated decision-making oversight through scoring thresholds. However, critical gaps significantly impact compliance: accuracy metrics and performance measurements are not declared in system documentation, formal risk assessment and management frameworks are not documented, and comprehensive operational control processes are absent from the reviewed materials."

EXAMPLE FORMAT for rationale (WITH legal references):
"The overall score reflects mixed compliance across five evaluated sub-requirements. Article 15 Para 1 (EU AI Act) received a score of 2, indicating the system's purpose is identified but lacks comprehensive robustness evidence. Article 15 Para 3 scored 0 as no accuracy metrics are declared, failing a mandatory EU AI Act requirement. ISO 42001 sections 6.1.1 and 8.1 both scored 1, showing minimal risk management and operational control evidence. Only section 6.1.2 achieved a score of 2, demonstrating some risk awareness without formal processes. These deficiencies in critical regulatory areas justify the low overall compliance score."

Respond in JSON format:
{{
    "score": "Integer 0-5 or 'N/A'",
    "auditor_notes": "Executive summary paragraph describing compliance status, evidence found, and specific gaps.",
    "rationale": "Detailed analytical justification referencing specific sub-requirement findings and their implications."
}}
"""

    def evaluate_sub_requirement(self, main_req_name: str, sub_req_name: str, source: str, regulatory_reference: str, associated_chunks: List[str]) -> Dict[str, Any]:
        """
        Evaluates a single sub-requirement using the LLM.
        """
        prompt = self._get_sub_prompt(main_req_name, sub_req_name, source, regulatory_reference, associated_chunks)
        # Question format that matches analytical response style
        ragas_question = f"What is the compliance status of sub-requirement '{sub_req_name}' from {source}?"

        response = self.llm.invoke(prompt).content.strip()
        try:
            cleaned = response.replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned)
        except Exception:
            result = {"score": "N/A", "rationale": "Parsing failed", "auditor_notes": response}

        result["ragas_question"] = ragas_question
        result["prompt"] = prompt  # Save prompt for logging
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
