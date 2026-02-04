import os
import re
import json
import yaml
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def process_law_text(raw_pages, source_name):
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
        
    # Semanthic separators for laws (Articles)
    custom_separators = [
        "\nArticle ", "\nArticolo ", 
        "\n\n", "\n", " ", ""
    ]
    
    splitter = RecursiveCharacterTextSplitter( # Initialize text splitter
        chunk_size=ingestion_params['chunk_size'], # Chunk size from params
        chunk_overlap=ingestion_params['chunk_overlap'], # Chunk overlap from params
        separators=custom_separators # Custom separators
    )
    
    all_docs = [] # To store all document chunks
    
    pdf_files = [f for f in os.listdir(raw_dir) if f.endswith('.pdf')] # List PDF files
    
    if not pdf_files:
        print(f"⚠ No PDF files found in {raw_dir}. Make sure you have run 'dvc pull'.")
        return

    for pdf_file in pdf_files:
        print(f"Analyzing {pdf_file}...")
        pdf_path = os.path.join(raw_dir, pdf_file) # Full path to PDF
        
        # Extract raw text from PDF by pages
        raw_pages = extract_text_from_pdf(pdf_path) # Extract text from PDF, divided by pages
        text = process_law_text(raw_pages, pdf_file) # Process raw text to maintain paragraph structure
        
        # Split text into chunks
        chunks = splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            all_docs.append({
                "source": pdf_file,
                "chunk_id": i,
                "content": chunk
            })
            
    # Save all chunks to a JSON file
    output_path = os.path.join(processed_dir, "chunks.json") # Output path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)
        
    print(f"✓ Ingestion completed. {len(all_docs)} chunks saved in {output_path}")

if __name__ == "__main__":
    main()
