"""  
Data Ingestion Script for Legal Documents
This script performs the following steps:
1. Loads parameters from params.yaml.
2. Reads PDF files from the raw data directory.
3. Extracts text from each PDF, maintaining paragraph structure.
4. Splits the text into chunks.
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
    

    # --- Custom ingestion for AI Act JSON ---

    # Produce ai_act_parsed.json from ai_act.html
    ai_act_html_path = os.path.join(raw_dir, "ai_act.html")
    if not os.path.exists(ai_act_html_path):
        print(f"⚠ AI Act HTML file not found at {ai_act_html_path}. Make sure you have run 'dvc pull'.")
    else:
        parse_ai_act_file_to_json(ai_act_html_path, os.path.join(processed_dir, "ai_act_parsed.json"))

        ai_act_json_path = os.path.join(processed_dir, "ai_act_parsed.json")

        # --- Custom ingestion for AI Act JSON ---
        if os.path.exists(ai_act_json_path):
            #print(f"Custom ingestion: found {ai_act_json_path}, processing AI Act...")
            with open(ai_act_json_path, "r", encoding="utf-8") as f:
                ai_act_sections = json.load(f)
            all_docs_aia = [] # Separate list for AI Act chunks to keep track of source
            for section in ai_act_sections:
                section_name = section.get("name", "unknown")
                section_type = section.get("type", "unknown")
                section_title = section.get("title", None)
                content = section.get("content", "")
                # Use the splitter directly on the full section content
                chunks = splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    all_docs_aia.append({
                        "source": f"ai_act::{section_name}",
                        "section_type": section_type,
                        "section_title": section_title,
                        "chunk_id": i,
                        "content": chunk
                    })
            print(f"✓ AI Act ingestion completed. {len(all_docs_aia)} chunks from AI Act.")

    # --- Custom ingestion for ISO PDF ---
    iso_pdf_path = os.path.join(raw_dir, "iso.pdf")
    if not os.path.exists(iso_pdf_path):
        print(f"⚠ ISO PDF file not found at {iso_pdf_path}. Make sure you have run 'dvc pull'.")
    else:
        parse_iso_file_to_json(iso_pdf_path, os.path.join(processed_dir, "iso_parsed.json"))
        iso_json_path = os.path.join(processed_dir, "iso_parsed.json")
        all_docs_iso = [] # Separate list for ISO chunks to keep track of source
        if os.path.exists(iso_json_path):   
            #print(f"Custom ingestion: found {iso_json_path}, processing ISO 42001...")
            with open(iso_json_path, "r", encoding="utf-8") as f:
                iso_sections = json.load(f)
            for section in iso_sections:
                section_id = section.get("section_id", "unknown")
                section_title = section.get("section_title", None)
                section_type = section.get("section_type", "unknown")
                content = section.get("content", "")
                # Use the splitter directly on the full section content
                chunks = splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    all_docs_iso.append({
                        "source": f"iso42001::{section_id}",
                        "section_type": section_type,
                        "section_title": section_title,
                        "chunk_id": i,
                        "content": chunk
                    })
            print(f"✓ ISO 42001 ingestion completed. {len(all_docs_iso)} chunks from ISO.")

    
    # Check if any PDF files are present for standard ingestion
    # Exclude AI Act and ISO since they are handled with custom parsers
    pdf_files = [f for f in os.listdir(raw_dir) if f.endswith('.pdf') and f not in ["ai_act.pdf", "iso.pdf"]]

    # --- Standard ingestion for all other PDFs ---
    all_docs_pdf = [] # Separate list for standard PDF chunks to keep track of source
    if not pdf_files and not all_docs:
        print(f"⚠ No other PDF files found in {raw_dir}.")
    
    else:
        for pdf_file in pdf_files:
            current_pdf_docs = []
            print(f"Analyzing {pdf_file}...")
            pdf_path = os.path.join(raw_dir, pdf_file)
            raw_pages = extract_text_from_pdf(pdf_path)
            text = process_law_text(raw_pages)
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                current_pdf_docs.append({
                    "source": pdf_file,
                    "chunk_id": i,
                    "content": chunk
                })
            all_docs_pdf.extend(current_pdf_docs)
            print(f"✓ {pdf_file} processed. {len(chunks)} chunks extracted.")

    # Combine custom ingestions with standard ones
    all_docs.extend(all_docs_aia) # Add AI Act chunks
    all_docs.extend(all_docs_iso) # Add ISO chunks
    if all_docs_pdf:
        all_docs.extend(all_docs_pdf) # Add standard PDF chunks

    # Save all chunks to a JSON file


    output_path = os.path.join(processed_dir, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)
    print(f"✓ Ingestion completed. {len(all_docs)} chunks saved in {output_path}")

if __name__ == "__main__":
    main()
