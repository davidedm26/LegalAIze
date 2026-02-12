from typing import Optional, List, Dict, Any
import fitz
import re
import json
from pathlib import Path

def ingest_iso_advanced(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        text = page.get_text()
        # Pulizia licenze e copyright
        text = re.sub(r"(Licenced to|BS ISO/IEC|ISO/IEC 2023|Page \d+|v\n|vi\n|vii\n).*", "", text)
        full_text += text

    parsed_data = []

    # --- 1. PARSING CLAUSE 4-10 (REQUISITI CORE) ---
    # Usiamo un lookahead negativo per evitare le righe con i puntini dell'indice
    # Cerchiamo: Nuova riga + Numero (4-10) + Punto + Numero + Spazio + Titolo
    # Ma ESCLUDIAMO se ci sono più di 3 puntini di fila nella riga
    clauses = re.finditer(
        r'\n(?![^\n]*\.{3,})([4-9]|10)\.(\d+(?:\.\d+)*)?\s+([^\n]+)\n(.*?)(?=\n(?![^\n]*\.{3,})(?:[4-9]|10)\.|\nAnnex|\Z)',
        full_text,
        re.DOTALL
    )
    
    for m in clauses:
        c_id = f"{m.group(1)}.{m.group(2)}" if m.group(2) else m.group(1)
        title = m.group(3).strip()
        content = m.group(4).strip()
        
        # Ulteriore pulizia per sicurezza: se il contenuto è troppo corto o il titolo ha puntini, scartiamo (è l'indice)
        if len(content) < 10 or "...." in title:
            continue

        parsed_data.append({
            "source": f"ISO 42001:2023::Clause {c_id}",
            "section_id": c_id,
            "section_title": title,
            "section_type": "management_requirement",
            "content": content,
            "metadata": {"normative": True, "annex": None}
        })

    # --- 2. PARSING ANNEX A (CONTROLLI) ---
    # Simile alle clausole, escludiamo le righe dell'indice con i puntini
    annex_a_matches = re.finditer(
        r'\n(?![^\n]*\.{3,})(A\.\d+(?:\.\d+)*)\s+([^\n]+)\n(.*?)(?=\n(?![^\n]*\.{3,})[A-B]\.|\nAnnex|\Z)', 
        full_text, 
        re.DOTALL
    )
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

    # --- 3. PARSING ANNEX B (GUIDA) ---
    annex_b_matches = re.finditer(
        r'\n(?![^\n]*\.{3,})(B\.\d+(?:\.\d+)*)\s+([^\n]+)\n(.*?)(?=\n(?![^\n]*\.{3,})B\.\d|\nAnnex [A-C]|\Z)', 
        full_text, 
        re.DOTALL
    )

    for m in annex_b_matches:
        b_id = m.group(1)
        mapping_target = b_id.replace('B', 'A')
        clean_title = re.split(r'\n|Control', m.group(2))[0].strip()
        
        raw_content = m.group(3).strip()
        parsed_data.append({
            "source": f"ISO 42001:2023::Annex B Guidance {b_id}",
            "section_id": b_id,
            "section_title": clean_title,
            "section_type": "implementation_guidance",
            "content": raw_content,
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
