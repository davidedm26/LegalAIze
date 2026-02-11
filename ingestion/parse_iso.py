from typing import Optional, List, Dict, Any
import fitz
import re
import json
from pathlib import Path

def ingest_iso_advanced(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        # Clean metadata and licenses to avoid noise in chunks
        text = page.get_text()
        text = re.sub(r"(Licenced to|BS ISO/IEC|ISO/IEC 2023|Page \d+|v\n|vi\n|vii\n).*", "", text)
        full_text += text

    parsed_data = []

    # --- 1. PARSING CLAUSE 4-10 (CORE REQUIREMENTS) ---
    # Identify sections like "4.1 Understanding the organization..."
    clauses = re.finditer(r'\n([4-9]|10)\.(\d+(?:\.\d+)*)?\s+([A-Z][a-zA-Z\s\-\,]+)\n(.*?)(?=\n\d+\.|\nAnnex|\Z)', full_text, re.DOTALL)
    for m in clauses:
        c_id = f"{m.group(1)}.{m.group(2)}" if m.group(2) else m.group(1)
        parsed_data.append({
            "source": f"ISO 42001:2023::Clause {c_id}",
            "section_id": c_id,
            "section_title": m.group(3).strip(),
            "section_type": "management_requirement",
            "content": m.group(4).strip(),
            "metadata": {"normative": True, "annex": None}
        })

    # --- 2. PARSING ANNEX A (NORMATIVE CONTROLS) ---
    # Control objectives start with A.x (e.g. A.2 Policies related to AI)
    # Specific controls start with A.x.x (e.g. A.2.2 AI policy)
    annex_a_matches = re.finditer(r'\n(A\.\d+(?:\.\d+)*)\s+([A-Z][a-zA-Z\s\-\,]+)\n(.*?)(?=\n[A-B]\.|\nAnnex|\Z)', full_text, re.DOTALL)
    for m in annex_a_matches:
        a_id = m.group(1)
        parsed_data.append({
            "source": f"ISO 42001:2023::Annex A Control {a_id}",
            "section_id": a_id,
            "section_title": m.group(2).strip(),
            "section_type": "control_requirement" if len(a_id.split('.')) > 2 else "control_objective",
            "content": m.group(3).strip(),
            "metadata": {"normative": True, "annex": "A"}
        })

    # --- 3. PARSING ANNEX B (IMPLEMENTATION GUIDANCE) ---
    # Each section in B maps 1:1 to a control in A (e.g. B.2.2 guides A.2.2)
    annex_b_matches = re.finditer(r'\n(B\.\d+(?:\.\d+)*)\s+([A-Z][a-zA-Z\s\-\,]+)\n(.*?)(?=\n[B-C]\.|\nAnnex|\Z)', full_text, re.DOTALL)
    for m in annex_b_matches:
        b_id = m.group(1)
        mapping_target = b_id.replace('B', 'A') # Semantic link to the control
        parsed_data.append({
            "source": f"ISO 42001:2023::Annex B Guidance {b_id}",
            "section_id": b_id,
            "section_title": m.group(2).strip(),
            "section_type": "implementation_guidance",
            "content": m.group(3).strip(),
            "metadata": {
                "normative": False, 
                "annex": "B",
                "maps_to_control": mapping_target
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
