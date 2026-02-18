"""  
Data Ingestion Script for Legal Documents
This script performs the following steps:
1. Loads parameters from params.yaml.
2. Reads PDF files from the raw data directory.
3. Extracts text from each PDF, maintaining paragraph structure.
4. Splits the text into chunks, adding enriched context such as section titles and metadata.
5. Saves the chunks as a JSON file in the processed data directory.

"""


import os
import re
import json
import yaml
from parse_aia import parse_ai_act_file_to_json  # Custom parser for AI Act HTML
from parse_iso import parse_iso_file_to_json  # Custom parser for ISO PDF

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


# --- Refactored Ingestion: Requirement-centric Extraction ---
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_ai_act_section(ref, ai_act_sections):
    """
    Extracts the full text for a given AI Act reference (e.g., 'Article 15 Para 1d', 'Annex XI Section 2').
    Handles paragraph/point extraction if specified.
    """
    ref = ref.strip()
    # Article extraction
    m = re.match(r"Article (\d+)(?:\s*Para\s*([\d\w/.,]+))?", ref, re.I)
    if m:
        art_num = m.group(1)
        para = m.group(2)
        for section in ai_act_sections:
            name = section.get("name", "").lower()
            content = section.get("content", "")
            if name == f"art_{art_num}":
                if para:
                    # Support for multiple paras/points (e.g. 1/3d)
                    paras = re.split(r"[ /,\.]+", para)
                    found = []
                    for p in paras:
                        p = p.strip()
                        if not p:
                            continue
                        # Paragraph extraction (e.g. 1.)
                        para_regex = rf"\n\s*{re.escape(p)}\."
                        matches = list(re.finditer(para_regex, content))
                        for match in matches:
                            start = match.end()
                            next_match = re.search(r"\n\s*[0-9a-zA-Z]+\." , content[start:])
                            end = start + next_match.start() if next_match else len(content)
                            found.append(content[start:end].strip())
                        # Point extraction (e.g. (d))
                        point_match = re.search(rf"\({p}\)[^\n]*", content)
                        if point_match:
                            found.append(point_match.group(0).strip())
                    if found:
                        return "\n".join(found)
                    else:
                        return content
                else:
                    return content
    # Annex extraction (e.g. 'Annex XI Section 2', 'Annex IV Para 2g')
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
                # Section extraction for Annex XI (and similar)
                if section_part:
                    section_num = re.findall(r"\d+", section_part)
                    if section_num:
                        # Find all Section headers
                        section_headers = list(re.finditer(r"Section\s*\d+", content, re.I))
                        # Find the requested section
                        for idx, header in enumerate(section_headers):
                            header_num = re.findall(r"\d+", header.group())
                            if header_num and header_num[0] == section_num[0]:
                                start = header.end()
                                # End at next section or end of content
                                end = section_headers[idx+1].start() if idx+1 < len(section_headers) else len(content)
                                section_content = content[start:end].strip()
                                # Para extraction
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
                                # Point extraction
                                if point_part:
                                    point_letter = re.findall(r"\w+", point_part)
                                    if point_letter:
                                        point_regex = rf"\({point_letter[0]}\)[^\n]*"
                                        point_match = re.search(point_regex, section_content)
                                        if point_match:
                                            return point_match.group(0).strip()
                                return section_content
                        # If section not found, fallback to full annex content
                        return content
                # Para extraction at annex level
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
                # Point extraction at annex level
                if point_part:
                    point_letter = re.findall(r"\w+", point_part)
                    if point_letter:
                        point_regex = rf"\({point_letter[0]}\)[^\n]*"
                        point_match = re.search(point_regex, content)
                        if point_match:
                            return point_match.group(0).strip()
                return content
    # Recital extraction
    if ref.lower().startswith("recital"):
        num = re.findall(r"\d+", ref)
        if num:
            name = f"rct {num[0]}"
            for section in ai_act_sections:
                if section.get("name", "").lower() == name:
                    return section.get("content", "")
    # Fallback: match in title or name
    for section in ai_act_sections:
        if ref.lower() in section.get("title", "").lower() or ref.lower() in section.get("name", "").lower():
            return section.get("content", "")
    return ""

def extract_iso_sections(ref, iso_sections):
    """
    For ISO: if ref is a section like '9.1', return all sections whose section_id starts with '9.1' (e.g. 9.1.1, 9.1.2, ...)
    Otherwise, match by section_id or in section_title.
    Returns a list of (section_id, section_title, content).
    """
    ref = ref.strip()
    results = []
    # Hierarchical match: e.g. 9.1 matches 9.1, 9.1.1, 9.1.2, ...
    if re.match(r"^\d+(\.\d+)+$", ref):
        for section in iso_sections:
            sid = section.get("section_id", "")
            if sid.startswith(ref):
                results.append((sid, section.get("section_title", ""), section.get("content", "")))
        return results
    # Exact match
    for section in iso_sections:
        if section.get("section_id", "").lower() == ref.lower():
            results.append((section.get("section_id", ""), section.get("section_title", ""), section.get("content", "")))
    if results:
        return results
    # Fallback: match in title
    for section in iso_sections:
        if ref.lower() in section.get("section_title", "").lower():
            results.append((section.get("section_id", ""), section.get("section_title", ""), section.get("content", "")))
    return results

def build_requirement_chunks(mapping, ai_act_sections, iso_sections):
    """
    For each requirement in mapping.json, build a chunk with full text for each referenced section (AI Act and ISO).
    Each chunk includes the title and the reference string.
    """
    requirement_chunks = []
    for principle in mapping.get("eu_ai_act_ethical_principle", []):
        ethical_principle = principle.get("ethical_principle", "")
        for req in principle.get("technical_requirements", []):
            req_name = req.get("name", "")
            id = req.get("id", "")
            eu_refs = req.get("eu_ai_act_articles", [])
            iso_refs = req.get("iso_42001_sections", [])
            eu_contents = []
            for ref in eu_refs:
                text = extract_ai_act_section(ref, ai_act_sections)
                eu_contents.append({
                    "reference": ref,
                    "content": f"[TITLE: {req_name}] [REF: {ref}]\n{text.strip()}"
                })
            iso_contents = []
            for ref in iso_refs:
                iso_texts = extract_iso_sections(ref, iso_sections)
                for sid, stitle, scontent in iso_texts:
                    iso_contents.append({
                        "reference": f"{ref} (matched: {sid})",
                        "content": f"[TITLE: {req_name}] [REF: {ref}] [ISO SECTION: {sid}] [TITLE: {stitle}]\n{scontent.strip()}"
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
    # Refactored logic: only requirement-centric extraction
    mapping_path = os.path.join("data", "mapping.json")
    processed_dir = "data/processed"

    parse_ai_act_file_to_json(
        filepath="data/raw_data/ai_act.html",
        output_path=os.path.join(processed_dir, "ai_act_parsed.json")
    )
    parse_iso_file_to_json(
        filepath="data/raw_data/iso.pdf",
        output_path=os.path.join(processed_dir, "iso_parsed.json")
    )

    ai_act_json_path = os.path.join(processed_dir, "ai_act_parsed.json")
    iso_json_path = os.path.join(processed_dir, "iso_parsed.json")
    requirement_output_path = os.path.join(processed_dir, "requirement_chunks.json")

    mapping = load_json(mapping_path)
    ai_act_sections = load_json(ai_act_json_path)
    iso_sections = load_json(iso_json_path)

    requirement_chunks = build_requirement_chunks(mapping, ai_act_sections, iso_sections)

    with open(requirement_output_path, "w", encoding="utf-8") as f:
        json.dump(requirement_chunks, f, indent=2, ensure_ascii=False)
    print(f"✓ requirement_chunks.json generated with {len(requirement_chunks)} requirement.")
    
main()
