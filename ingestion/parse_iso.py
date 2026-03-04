
"""
ISO 42001:2023 PDF Parser

This module extracts structured sections from the ISO/IEC 42001:2023 standard PDF,
converting the management system requirements into a machine-readable JSON format.

Output Format:
Each extracted section is a dictionary containing:
{
  "name": "Clause 6.1.1",
  "section_id": "6.1.1",
  "section_title": "6.1 Planning - 6.1.1 Actions to address risks",
  "section_type": "management_requirement" | "implementation_guidance",
  "content": "The organization shall...",          # For normative clauses
  "control": "AI system inventory shall...",    # For Annex B controls
  "implementation_guidance": "Consider...",     # For Annex B guidance
  "metadata": {
    "normative": true,
    "annex": null | "B"
  }
}

"""

from typing import Optional, List, Dict, Any
import fitz
import re
import json
from pathlib import Path

def clean_chunk(text: str) -> str:
    # Remove copyright marks and page artifacts
    text = re.sub(r'\n\s*©.*?(ISO/IEC 42001:2023\(E\))?', '', text, flags=re.DOTALL)
    text = re.sub(r'\n\s*©.*?\n', '', text, flags=re.DOTALL)
    text = re.sub(r'\n\s*\d+\n*ISO/IEC 42001:2023\(E\)', '', text, flags=re.DOTALL)
    # Remove NOTE blocks (from NOTE to next newline or end)
    text = re.sub(r'NOTE[\s\S]*?(?=\n[A-Z]|\n\d|\n\s*©|\n\s*\-|\n\s*\*|\n\s*\Z)', '', text, flags=re.IGNORECASE)
    # Remove isolated page numbers and artifacts
    text = re.sub(r'\n\s*\d+\n', '\n', text)
    # Remove excessive newlines (but preserve bullet lists)
    text = re.sub(r'\n{3,}', '\n', text)
    # Remove leading/trailing whitespace and newlines
    text = text.strip()
    return text

def ingest_iso_advanced(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        text = page.get_text()
        # Cleaning licenses and copyright
        text = re.sub(r"(Licenced to|BS ISO/IEC|ISO/IEC 2023|Page \d+|v\n|vi\n|vii\n).*", "", text)
        full_text += text

    parsed_data = []

    # --- 1. PARSING CLAUSE 4-10 (CORE REQUIREMENTS) ---
    # Build a map of level 2 section titles and their content (robust boundaries)
    level2_map = {}
    level2_matches = list(re.finditer(
        r'^([4-9]|10)\.(\d+)\s+([A-Z][^\n]+)$',
        full_text,
        re.MULTILINE
    ))
    # Add a sentinel for the end of the document
    doc_end = len(full_text)
    for idx, m in enumerate(level2_matches):
        sec_id = f"{m.group(1)}.{m.group(2)}"
        sec_title = m.group(3).strip()
        start = m.end()
        if idx + 1 < len(level2_matches):
            end = level2_matches[idx + 1].start()
        else:
            # Stop at next Annex or end of file
            annex_match = re.search(r'\nAnnex', full_text[start:], re.DOTALL)
            end = start + annex_match.start() if annex_match else doc_end
        content = full_text[start:end].strip()
        # Truncate if the next line is a chapter heading (e.g., '5 Leadership')
        chapter_heading_match = re.search(r'^(\d+)\s+[A-Z][^\n]+', content, re.MULTILINE)
        if chapter_heading_match:
            # Only keep content before the chapter heading
            content = content[:chapter_heading_match.start()].strip()
        level2_map[sec_id] = sec_title
        # Aggiungi tutte le sezioni di livello 2 come chunk, tranne quelle speciali
        special_level3 = {'6.1', '7.5', '9.2', '9.3'}
        if sec_id not in special_level3:
            if len(content) < 10 or "...." in sec_title:
                continue
            parsed_data.append({
                "name": f"Clause {sec_id}",
                "section_id": sec_id,
                "section_title": f"{sec_id} - {sec_title}",
                "section_type": "management_requirement",
                "content": clean_chunk(content),
                "metadata": {"normative": True, "annex": None}
            })

    # Gestisci solo i chunk di livello 3 per 6.1 e 7.5
    leaf_regex = re.finditer(
        r'\n(?![^\n]*\.{3,})([4-9]|10)\.(\d+)\.(\d+)\s+([^\n]+)\n(.*?)(?=\n(?![^\n]*\.{3,})([4-9]|10)\.(\d+)\.(\d+)|\nAnnex|\Z)',
        full_text,
        re.DOTALL
    )
    for m in leaf_regex:
        sec_id = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
        parent_id = f"{m.group(1)}.{m.group(2)}"
        parent_title = level2_map.get(parent_id, "")
        sec_title = m.group(4).strip()
        content = m.group(5).strip()
        combined_title = f"{sec_id} {parent_title} - {sec_title}".strip()
        # Aggiungi solo se parent_id è tra le speciali
        if parent_id in special_level3:
            if len(content) < 10 or "...." in sec_title:
                continue
            parsed_data.append({
                "name": f"Clause {sec_id}",
                "section_id": sec_id,
                "section_title": combined_title,
                "section_type": "management_requirement",
                "content": clean_chunk(content),
                "metadata": {"normative": True, "annex": None}
            })

    # --- 2. PARSING ANNEX A (CONTROLS) ---
    # Removed   
    
    # --- 3. PARSING ANNEX B (GUIDANCE) ---
    annex_b_matches = re.finditer(
        r'\n(?![^\n]*\.{3,})(B\.\d+(?:\.\d+)*)\s+([^\n]+)\n(.*?)(?=\n(?![^\n]*\.{3,})B\.\d|\nAnnex [A-C]|\Z)', 
        full_text, 
        re.DOTALL
    )
    for m in annex_b_matches:
        b_id = m.group(1)
        mapping_target = b_id.replace('B', 'A')
        parent_id = '.'.join(b_id.split('.')[:2])
        parent_title = level2_map.get(parent_id, "Annex B")
        clean_title = re.split(r'\n|Control', m.group(2))[0].strip()
        raw_content = m.group(3).strip()
        combined_title = f"{b_id} {parent_title} - {clean_title}".strip()
        # Extract 'Control' and 'Implementation guidance' blocks
        control_match = re.search(r'Control\n(.*?)(?=Implementation guidance|\nOther information|\nNOTE|\Z)', raw_content, re.DOTALL)
        guidance_match = re.search(r'Implementation guidance\n(.*?)(?=\nOther information|\nNOTE|\Z)', raw_content, re.DOTALL)
        if control_match and guidance_match:
            control_text = control_match.group(1).strip()
            guidance_text = guidance_match.group(1).strip()
            parsed_data.append({
                "name": f"Annex B Guidance {b_id}",
                "section_id": b_id,
                "section_title": combined_title,
                "section_type": "implementation_guidance",
                "control": clean_chunk(control_text),
                "implementation_guidance": clean_chunk(guidance_text),
                "metadata": {
                    "normative": False, 
                    "annex": "B"
                }
            })
    return parsed_data


def parse_iso_file_to_json(filepath: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Main function: reads the ISO PDF file, parses sections, and optionally saves to JSON.
    Args:
        filepath (str): Path to the PDF file.
        output_path (Optional[str]): Path to save the JSON file (if provided).
    Returns:
        List[Dict[str, Any]]: List of extracted section objects.
    """
    results = ingest_iso_advanced(filepath)
    if output_path:
        out_path = Path(output_path)
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parse_iso_file_to_json(
        filepath="data/raw_data/iso.pdf",
        output_path="data/processed/iso_parsed.json"
    )