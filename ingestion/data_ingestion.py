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
import fitz  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from parse_aia import parse_ai_act_file_to_json  # Custom parser for AI Act HTML
from parse_iso import parse_iso_file_to_json  # Custom parser for ISO PDF

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file and returns a list of dictionaries (page, text)."""
    doc = fitz.open(pdf_path)
    text_content = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_content.append({
            "page": page_num + 1,
            "text": page.get_text()
        })
    return text_content

def process_law_text(raw_pages):
    """
    Merges pages while maintaining paragraph structure.
    """
    full_text = ""
    for p in raw_pages: # for each page add its text
        full_text += f"\n{p['text']}"
    
    # Cleaning text
    full_text = re.sub(r'[ \t]+', ' ', full_text) # replace multiple spaces/tabs with single space
    full_text = re.sub(r'\n\s*\n+', '\n\n', full_text) # keep double line breaks for paragraphs
    
    return full_text.strip() # return cleaned text

def main():
    params = load_params() # Load parameters from params.yaml
    ingestion_params = params['ingestion'] # Ingestion parameters
    
    raw_dir = ingestion_params['raw_data_dir'] # Directory with raw PDFs
    processed_dir = ingestion_params['processed_data_dir'] # Directory for processed data
    
    if not os.path.exists(processed_dir): # Create processed data directory if not exists
        os.makedirs(processed_dir)
        
    # Semantic separators for laws (Articles)
    custom_separators = [
        "\n\n", "\n", " ", ""
    ]
    
    splitter = RecursiveCharacterTextSplitter( # Initialize text splitter
        chunk_size=ingestion_params['chunk_size'], # Chunk size from params
        chunk_overlap=ingestion_params['chunk_overlap'], # Chunk overlap from params
        separators=custom_separators # Custom separators
    )
    
    all_docs = [] # To store all document chunks
    all_docs_aia = [] # Separate list for AI Act chunks
    all_docs_iso = [] # Separate list for ISO chunks
    

    # --- Custom ingestion for AI Act  ---
    ai_act_html_path = os.path.join(raw_dir, "ai_act.html")
    if not os.path.exists(ai_act_html_path):
        print(f"⚠ AI Act HTML file not found at {ai_act_html_path}. Make sure you have run 'dvc pull'.")
    else:
        parse_ai_act_file_to_json(ai_act_html_path, os.path.join(processed_dir, "ai_act_parsed.json")) # Parse AI Act HTML and save as JSON
        ai_act_json_path = os.path.join(processed_dir, "ai_act_parsed.json")
        if os.path.exists(ai_act_json_path):
            with open(ai_act_json_path, "r", encoding="utf-8") as f:
                ai_act_sections = json.load(f)
            for section in ai_act_sections: # Process each section of AI Act and create enriched chunks
                s_name = section.get("name", "unknown")
                s_type = section.get("type", "unknown")
                s_title = section.get("title", "No Title")
                annex = section.get("annex", "")
                content = section.get("content", "")

                chunks = splitter.split_text(content) # Split content into chunks using the text splitter

                for i, chunk in enumerate(chunks): # Add enriched context to each chunk (e.g. section name, type, title, annex reference)
                    header = f"[SOURCE : AI ACT]"
                    if annex:
                        header += f" [ANNEX {annex}]"
                    header += f" [PART {i+1}/{len(chunks)}]"
                    header += f" [TYPE: {s_type.upper()}] [NAME: {s_name}] [TITLE: {s_title}]\n"
                    enriched_content = header + chunk
                    all_docs_aia.append({
                        "source": f"ai_act::{s_name}",
                        "section_type": s_type,
                        "section_title": s_title,
                        "chunk_id": i,
                        "content": enriched_content
                    })
            print(f"✓ AI Act ingestion completed. {len(all_docs_aia)} enriched chunks.") 

    # --- Custom ingestion for ISO  ---
    iso_pdf_path = os.path.join(raw_dir, "iso.pdf")
    if not os.path.exists(iso_pdf_path):
        print(f"⚠ ISO PDF file not found at {iso_pdf_path}. Make sure you have run 'dvc pull'.")
    else:
        parse_iso_file_to_json(iso_pdf_path, os.path.join(processed_dir, "iso_parsed.json")) # Parse ISO PDF and save as JSON
        iso_json_path = os.path.join(processed_dir, "iso_parsed.json")
        if os.path.exists(iso_json_path):
            with open(iso_json_path, "r", encoding="utf-8") as f:
                iso_sections = json.load(f)

            annex_a_map = {}
            for section in iso_sections: 
                # Annex A and Annex B are strongly linked, so we create a mapping of Annex A controls to enrich Annex B chunks
                if section.get("section_type") == "control_requirement" or section.get("section_type") == "control_objective": 
                    annex_a_map[section.get("section_id")] = section.get("content", "")

                s_id = section.get("section_id", "unknown")
                s_title = section.get("section_title", "No Title")
                s_type = section.get("section_type", "unknown")
                content = section.get("content", "")
                
                if s_type == "management_requirement": # Remaining sections
                    chunks = splitter.split_text(content)
                    for i, chunk in enumerate(chunks):
                        header = f"[SOURCE : ISO 42001] [ID: {s_id}] [TYPE: {s_type.upper()}] [Part {i+1}/{len(chunks)}] [TITLE: {s_title}]\n"
                        enriched_content = header + chunk
                        all_docs_iso.append({
                            "source": f"iso42001::{s_id}",
                            "section_type": s_type,
                            "section_title": s_title,
                            "chunk_id": i,
                            "content": enriched_content
                        })
                elif s_type == "implementation_guidance": # Annex B sections
                    
                    mapping_target = section.get("metadata", {}).get("maps_to_control", "")
                    annex_a_content = annex_a_map.get(mapping_target, "")
                    chunks = splitter.split_text(content)

                    # Add enriched context to each chunk, including reference to mapped Annex A control and its content if available (since Annex B is guidance for Annex A controls)
                    for i, chunk in enumerate(chunks):
                        header = f"[SOURCE : ISO 42001] [ID: {s_id}] [TYPE: {s_type.upper()}] [Part {i+1}/{len(chunks)}] [TITLE: {s_title}]"
                        if mapping_target:
                            header += f" [ANNEX A REF: {mapping_target}]"
                        if annex_a_content:
                            header += f" [ CORRISPONDENT ANNEX A CONTENT: {annex_a_content}...]"
                        header += "\n"
                        enriched_content = header + chunk
                        all_docs_iso.append({
                            "source": f"iso42001::{s_id}",
                            "section_type": s_type,
                            "section_title": s_title,
                            "chunk_id": i,
                            "content": enriched_content
                        })
            print(f"✓ ISO 42001 ingestion completed. {len(all_docs_iso)} enriched chunks.")


    # Combine custom ingestions
    all_docs.extend(all_docs_aia) # Add AI Act chunks
    all_docs.extend(all_docs_iso) # Add ISO chunks


    # Save all chunks to a JSON file
    output_path = os.path.join(processed_dir, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)
    print(f"✓ Ingestion completed. {len(all_docs)} chunks saved in {output_path}")

if __name__ == "__main__":
    main()
