"""
Legal Document Ingestion Pipeline

This module orchestrates the end-to-end ingestion process for regulatory documents:

Pipeline Steps:
1. Parse raw legal documents (AI Act HTML, ISO 42001 PDF) into structured JSON
2. Load requirement mapping that links ethical principles to regulatory references
3. Extract relevant sections from parsed documents based on requirement mapping
4. Generate requirement-centric chunks with associated regulatory content
5. Save chunks as JSON for downstream vectorization and RAG retrieval

Key Components:
- AI Act Parser: Extracts articles, paragraphs, points, annexes, and recitals
- ISO Parser: Extracts sections, controls, and implementation guidance
- Requirement Mapper: Links technical requirements to specific regulatory references
- Chunk Generator: Creates contextualized chunks for each compliance requirement

Output:
- requirement_chunks.json: Structured JSON with {requirement → regulatory chunks} mapping
"""

import os
import re
import json
import yaml
from typing import List, Dict, Any, Tuple
from parse_aia import parse_ai_act_file_to_json
from parse_iso import parse_iso_file_to_json


def load_params() -> Dict[str, Any]:
    """Load configuration parameters from params.yaml."""
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Any:
    """Load and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_ai_act_section(ref: str, ai_act_sections: List[Dict[str, Any]]) -> str:
    """
    Extract the full text for a given AI Act reference.
    
    Supports extraction of:
    - Articles with optional paragraphs/points (e.g., 'Article 15', 'Article 15 Para 1d')
    - Annexes with optional sections/paragraphs (e.g., 'Annex XI Section 2', 'Annex IV Para 2g')
    - Recitals (e.g., 'Recital 47')
    
    Args:
        ref: Reference string (e.g., 'Article 15 Para 1d', 'Annex XI Section 2')
        ai_act_sections: List of parsed AI Act sections
    
    Returns:
        Extracted text content, or empty string if not found
    
    Examples:
        >>> extract_ai_act_section("Article 15 Para 1", ai_act_sections)
        "1. High-risk AI systems shall be designed..."
        
        >>> extract_ai_act_section("Annex XI Section 2", ai_act_sections)
        "Section 2\nDocumentation of risk management..."
    """
    ref = ref.strip()
    
    # -------------------------------------------------------------------------
    # Article Extraction (e.g., 'Article 15', 'Article 15 Para 1d')
    # -------------------------------------------------------------------------
    m = re.match(r"Article (\d+)(?:\s*Para\s*([\d\w/.,]+))?", ref, re.I)
    if m:
        art_num = m.group(1)
        para = m.group(2)
        for section in ai_act_sections:
            name = section.get("name", "").lower()
            content = section.get("content", "")
            if name == f"art_{art_num}":
                if para:
                    # Support multiple paragraphs/points separated by / , . (e.g., '1/3d')
                    paras = re.split(r"[ /,\.]+", para)
                    found = []
                    for p in paras:
                        p = p.strip()
                        if not p:
                            continue
                        # Extract paragraph by number (e.g., '1.')
                        para_regex = rf"\n\s*{re.escape(p)}\."
                        matches = list(re.finditer(para_regex, content))
                        for match in matches:
                            start = match.end()
                            next_match = re.search(r"\n\s*[0-9a-zA-Z]+\." , content[start:])
                            end = start + next_match.start() if next_match else len(content)
                            found.append(content[start:end].strip())
                        # Extract point by letter (e.g., '(d)')
                        point_match = re.search(rf"\({p}\)[^\n]*", content)
                        if point_match:
                            found.append(point_match.group(0).strip())
                    if found:
                        return "\n".join(found)
                    else:
                        return content
                else:
                    return content
    
    # -------------------------------------------------------------------------
    # Annex Extraction (e.g., 'Annex XI Section 2', 'Annex IV Para 2g')
    # -------------------------------------------------------------------------
    annex_match = re.match(r"Annex ([A-Z]+)\s*(Section\s*\d+)?\s*(Para\s*[\d/\w]+)?\s*(\([\w]+\))?", ref, re.I)
    if annex_match:
        annex_id = annex_match.group(1)
        section_part = annex_match.group(2)
        para_part = annex_match.group(3)
        point_part = annex_match.group(4)
        for section in ai_act_sections:
            annex_name = section.get("name", "").lower()
            annex_title = section.get("title", "").lower()
            annex_field = section.get("annex", "").lower()
            if f"anx_{annex_id.lower()}" == annex_name or f"annex {annex_id.lower()}" in annex_title or f"annex {annex_id.lower()}" in annex_field:
                content = section.get("content", "")
                # Extract specific section within annex (e.g., 'Section 2' in Annex XI)
                if section_part:
                    section_num = re.findall(r"\d+", section_part)
                    if section_num:
                        # Locate all section headers in the annex
                        section_headers = list(re.finditer(r"Section\s*\d+", content, re.I))
                        # Find and extract the requested section number
                        for idx, header in enumerate(section_headers):
                            header_num = re.findall(r"\d+", header.group())
                            if header_num and header_num[0] == section_num[0]:
                                start = header.end()
                                # Section ends at next section header or end of document
                                end = section_headers[idx+1].start() if idx+1 < len(section_headers) else len(content)
                                section_content = content[start:end].strip()
                                # Extract paragraph within the section
                                if para_part:
                                    para_nums = re.findall(r"[\d\w]+", para_part)
                                    found = []
                                    for p in para_nums:
                                        para_regex = rf"\n\s*{p}\.[^\n]*"
                                        para_match = re.search(para_regex, section_content)
                                        if para_match:
                                            found.append(para_match.group(0).strip())
                                        point_match = re.search(rf"\({p}\)[^\n]*", section_content)
                                        if point_match:
                                            found.append(point_match.group(0).strip())
                                    if found:
                                        return "\n".join(found)
                                    else:
                                        return section_content
                                # Extract point within the section
                                if point_part:
                                    point_letter = re.findall(r"\w+", point_part)
                                    if point_letter:
                                        point_regex = rf"\({point_letter[0]}\)[^\n]*"
                                        point_match = re.search(point_regex, section_content)
                                        if point_match:
                                            return point_match.group(0).strip()
                                return section_content
                        # Fallback: return full annex if section not found
                        return content
                # Extract paragraph at annex level (without section specification)
                if para_part:
                    para_nums = re.findall(r"[\d\w]+", para_part)
                    found = []
                    for p in para_nums:
                        para_regex = rf"\n\s*{p}\.[^\n]*"
                        para_match = re.search(para_regex, content)
                        if para_match:
                            found.append(para_match.group(0).strip())
                        point_match = re.search(rf"\({p}\)[^\n]*", content)
                        if point_match:
                            found.append(point_match.group(0).strip())
                    if found:
                        return "\n".join(found)
                    else:
                        return content
                # Extract point at annex level (without section specification)
                if point_part:
                    point_letter = re.findall(r"\w+", point_part)
                    if point_letter:
                        point_regex = rf"\({point_letter[0]}\)[^\n]*"
                        point_match = re.search(point_regex, content)
                        if point_match:
                            return point_match.group(0).strip()
                return content
    
    # -------------------------------------------------------------------------
    # Recital Extraction (e.g., 'Recital 47')
    # -------------------------------------------------------------------------
    if ref.lower().startswith("recital"):
        num = re.findall(r"\d+", ref)
        if num:
            name = f"rct {num[0]}"
            for section in ai_act_sections:
                if section.get("name", "").lower() == name:
                    return section.get("content", "")
    
    # Fallback: Fuzzy match in section title or name
    for section in ai_act_sections:
        if ref.lower() in section.get("title", "").lower() or ref.lower() in section.get("name", "").lower():
            return section.get("content", "")
    return ""

def extract_iso_sections(ref: str, iso_sections: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """
    Extract ISO 42001 sections matching the given reference.
    
    Extraction Strategy:
    - Exact match on section_id (e.g., '9.1' matches only '9.1', not '9.1.1')
    - Returns all matching sections with their metadata
    
    Args:
        ref: ISO section reference (e.g., '9.1', 'B.3.2')
        iso_sections: List of parsed ISO sections
    
    Returns:
        List of tuples: (section_id, section_title, content)
    
    Examples:
        >>> extract_iso_sections("9.1", iso_sections)
        [('9.1', 'Leadership and commitment', 'Top management shall...')]
        
        >>> extract_iso_sections("B.3.2", iso_sections)
        [('B.3.2', 'AI system inventory', control_text, guidance_text)]
    """
    ref = ref.strip()
    results = []
    # Exact match on section_id (no fuzzy matching)
    for section in iso_sections:
        if section.get("section_id", "").lower() == ref.lower():
            results.append((section.get("section_id", ""), section.get("section_title", ""), section.get("content", "")))
    return results

def collect_chunks_for_requirement(
    mapping: Dict[str, Any], 
    ai_act_sections: List[Dict[str, Any]], 
    iso_sections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate requirement-centric chunks by linking requirements to their regulatory sources.
    
    Process:
    1. Iterate through ethical principles and technical requirements in mapping
    2. For each requirement, extract referenced AI Act articles and ISO sections
    3. Structure chunks with requirement metadata and regulatory content
    4. Handle special cases (ISO Annex B controls with implementation guidance)
    
    Args:
        mapping: Requirement mapping linking ethical principles to regulatory references
        ai_act_sections: Parsed AI Act sections
        iso_sections: Parsed ISO 42001 sections
    
    Returns:
        List of requirement chunks, each containing:
        - id: Requirement identifier (e.g., 'HUM_AGENCY_01')
        - ethicalPrinciple: Ethical principle category
        - requirementName: Technical requirement name
        - euAiActArticles: List of AI Act articles with content
        - iso42001Reference: List of ISO sections with content/controls
    
    Example Output:
        [
          {
            "id": "HUM_AGENCY_01",
            "ethicalPrinciple": "Human Agency & Oversight",
            "requirementName": "Roles and Responsibilities",
            "euAiActArticles": [{"reference": "Article 26", "content": "..."}],
            "iso42001Reference": [{"reference": "5.3", "content": "..."}]
          }
        ]
    """

    requirement_chunks = []
    for principle in mapping.get("eu_ai_act_ethical_principle", []):
        ethical_principle = principle.get("ethical_principle", "")
        for req in principle.get("technical_requirements", []):
            req_name = req.get("name", "")
            id = req.get("id", "")
            eu_refs = req.get("eu_ai_act_articles", [])
            iso_refs = req.get("iso_42001_sections", [])
            
            # Extract AI Act articles
            eu_contents = []
            for ref in eu_refs:
                text = extract_ai_act_section(ref, ai_act_sections)
                eu_contents.append({
                    "reference": ref,
                    "content": text.strip()
                })
            
            # Extract ISO 42001 sections
            iso_contents = []
            for ref in iso_refs:
                iso_texts = extract_iso_sections(ref, iso_sections)
                for sid, stitle, scontent in iso_texts:
                    # Find the full section object for Annex B controls
                    section_obj = next((s for s in iso_sections if s.get("section_id", "") == sid), None)
                    # Special handling for Annex B controls (split control + guidance)
                    if sid.startswith("B.") and section_obj:
                        iso_contents.append({
                            "reference": sid,
                            "control": section_obj.get("control", ""),
                            "implementation_guidance": section_obj.get("implementation_guidance", "")
                        })
                    else:
                        # Standard sections with title and content
                        iso_contents.append({
                            "reference": sid,
                            "content": f"[TITLE: {stitle}]\n{scontent.strip()}"
                        })
            requirement_chunks.append({
                "id": id,
                "ethicalPrinciple": ethical_principle,
                "requirementName": req_name,
                "euAiActArticles": eu_contents,
                "iso42001Reference": iso_contents
            })
    return requirement_chunks

def main():
    """
    Main ingestion pipeline execution.
    
    Pipeline Stages:
    1. Parse raw documents (AI Act HTML, ISO PDF) → structured JSON
    2. Load requirement mapping and parsed documents
    3. Extract and link regulatory chunks to requirements
    4. Save requirement_chunks.json for vectorization
    
    Output Files:
    - data/processed/ai_act_parsed.json: Structured AI Act sections
    - data/processed/iso_parsed.json: Structured ISO 42001 sections
    - data/processed/requirement_chunks.json: Requirement-to-regulatory mapping
    """
    mapping_path = os.path.join("data", "mapping.json")
    processed_dir = os.path.join("data", "processed")
    
    # -------------------------------------------------------------------------
    # Stage 1: Parse Raw Legal Documents
    # ------------------------------------------------------------------------- 
    parse_ai_act_file_to_json(
        filepath="data/raw_data/ai_act.html",
        output_path=os.path.join(processed_dir, "ai_act_parsed.json")
    )
    parse_iso_file_to_json(
        filepath="data/raw_data/iso.pdf",
        output_path=os.path.join(processed_dir, "iso_parsed.json")
    )
    
    # -------------------------------------------------------------------------
    # Stage 2: Load Structured Data
    # -------------------------------------------------------------------------
    ai_act_json_path = os.path.join(processed_dir, "ai_act_parsed.json")
    iso_json_path = os.path.join(processed_dir, "iso_parsed.json")
    requirement_output_path = os.path.join(processed_dir, "requirement_chunks.json")

    mapping = load_json(mapping_path)
    ai_act_sections = load_json(ai_act_json_path)
    iso_sections = load_json(iso_json_path)
    
    # -------------------------------------------------------------------------
    # Stage 3: Generate Requirement-Centric Chunks
    # -------------------------------------------------------------------------
    requirement_chunks = collect_chunks_for_requirement(mapping, ai_act_sections, iso_sections)

    # -------------------------------------------------------------------------
    # Stage 4: Save Output
    # -------------------------------------------------------------------------
    with open(requirement_output_path, "w", encoding="utf-8") as f:
        json.dump(requirement_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Ingestion complete: {len(requirement_chunks)} requirements processed")
    print(f"  → {ai_act_json_path}")
    print(f"  → {iso_json_path}")
    print(f"  → {requirement_output_path}")


if __name__ == "__main__":
    main()
