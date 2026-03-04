"""
This module defines the EvaluationEngine class, which is responsible for evaluating compliance of documents against regulatory requirements using a language model (LLM). It provides methods to evaluate individual sub-requirements and aggregate results into a final compliance assessment. The engine generates detailed prompts for the LLM to ensure structured and comprehensive responses, including both technical rationales and executive summaries for stakeholders.
"""

import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI

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
{chr(10).join(relevant_chunks) if relevant_chunks else "[NO RELEVANT DOCUMENT CHUNKS FOUND]"}

INSTRUCTIONS:
1. Analyze the chunks carefully. Look for ANY explicit mentions OR implicit evidence that addresses the regulatory context.
2. Be objective. If the chunks provide partial or related evidence, explain how it relates to the requirement rather than just saying "no evidence".
3. Support your reasoning by referencing specific parts of the text (e.g., "The document states that...").
4. If NO CHUNKS are provided, you MUST return 'N/A' and explain that evaluation is not applicable due to missing document content.
5. Score 0 if chunks exist but are completely irrelevant or contain no information related to the regulatory requirement.
6. Use 'N/A' only when the evaluation cannot be performed (no chunks). Use 0 when evaluation is performed but finds no evidence.

Respond in JSON format:
{{
    "rationale": "Detailed explanation of findings referencing the provided text. If no chunks provided, state: 'Evaluation not applicable - no document content available for analysis.'",
    "score": "Integer 0-5 (0=No evidence found in document, 5=Fully compliant) or 'N/A' (evaluation not applicable)",
    "auditor_notes": "Concise summary for the final report."
}}
"""

    def _get_aggregate_prompt(self, sub_results: List[Dict[str, Any]], computed_score) -> str:
        simplified_results = []
        for res in sub_results:
            simplified_results.append({
                "reference": res.get("reference"),
                "source": res.get("source"),
                "score": res.get("score"),
                "auditor_notes": res.get("auditor_notes", res.get("answer", "")) # Use concise auditor_notes
            })

        # Format computed score (handle both numeric and N/A)
        score_display = computed_score if isinstance(computed_score, str) else f"{computed_score:.1f}"
        
        return f"""
You are a Lead AI Auditor consolidating findings for a main requirement based on several sub-requirement evaluations.

SUB-REQUIREMENT FINDINGS:
{json.dumps(simplified_results, indent=2, ensure_ascii=False)}

COMPUTED OVERALL SCORE: {score_display}
(This score is the arithmetic mean of numeric sub-requirement scores. N/A scores are excluded from the average. If ALL sub-requirements are N/A, the overall score is also N/A.)

TASK:
Aggregate these findings into a final compliance assessment for the main requirement.

INSTRUCTIONS:
1. If the computed score is N/A (all sub-requirements are N/A):
   - auditor_notes: "Evaluation not applicable: insufficient or improperly formatted documentation was provided for compliance assessment."
   - rationale: Explain that all sub-requirements returned N/A due to lack of analyzable content.

2. Write 'auditor_notes' as an EXECUTIVE SUMMARY for management/stakeholders (NO technical legal references):
   - Write in paragraph form (NOT a list, NO bullet points, NO semicolons separating items)
   - Start with overall compliance status (e.g., "Partially compliant", "Non-compliant", "Fully compliant", "Evaluation not applicable")
   - Describe FUNCTIONALLY what was found and what gaps exist
   - Use business-friendly language: "accuracy measurements", "risk management processes", "oversight mechanisms"
   - DO NOT mention specific article numbers, paragraph numbers, or section codes
   - Focus on impact and actionable insights
   - Note any N/A sub-requirements as "could not be evaluated due to insufficient documentation"

3. Write 'rationale' as a TECHNICAL ANALYSIS for compliance experts (WITH legal references):
   - Reference specific sub-requirement codes and scores (e.g., "Article 15 Para 1 scored 2", "ISO 42001 section 6.1.1 scored 1")
   - For N/A scores, note: "[Reference] received N/A (evaluation not applicable due to insufficient document content)"
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

Respond in JSON format (DO NOT include 'score' - it has already been calculated):
{{
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
        Score is computed as the arithmetic mean of numeric sub-requirement scores.
        Returns N/A if all sub-requirements are N/A (evaluation not applicable).
        """
        # Calculate average score from sub-requirements
        numeric_scores = []
        na_count = 0
        for res in sub_results:
            score = res.get("score")
            if isinstance(score, str) and score.upper() == "N/A":
                na_count += 1
            else:
                try:
                    numeric_scores.append(float(score))
                except (ValueError, TypeError):
                    pass  # Skip non-numeric scores
        
        # If all sub-requirements are N/A, return N/A for the main requirement
        if na_count == len(sub_results) and na_count > 0:
            computed_score = "N/A"
        elif numeric_scores:
            computed_score = sum(numeric_scores) / len(numeric_scores)
        else:
            computed_score = 0.0
        
        agg_prompt = self._get_aggregate_prompt(sub_results, computed_score)
        agg_response = self.llm.invoke(agg_prompt).content.strip()
        try:
            cleaned_agg = agg_response.replace("```json", "").replace("```", "").strip()
            agg_result = json.loads(cleaned_agg)
        except Exception:
            agg_result = {"auditor_notes": "Parsing failed", "rationale": agg_response}

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

        # Handle score formatting (keep N/A as string, round numeric scores)
        final_score = computed_score if isinstance(computed_score, str) else round(computed_score, 1)
        
        return {
            "score": final_score,
            "auditor_notes": notes_str,
            "rationale": agg_result.get("rationale", ""),
            "prompt": agg_prompt
        }
