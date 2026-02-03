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
    """Estrae testo da un file PDF e restituisce una lista di dizionari (pagina, testo)."""
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
    Unisce le pagine mantenendo la struttura dei paragrafi.
    """
    full_text = ""
    for p in raw_pages:
        full_text += f"\n{p['text']}"
    
    # Pulizia: riduciamo gli spazi ma manteniamo i ritorni a capo (\n)
    full_text = re.sub(r'[ \t]+', ' ', full_text)
    full_text = re.sub(r'\n\s*\n+', '\n\n', full_text)
    
    return full_text.strip()

def main():
    params = load_params()
    ingestion_params = params['ingestion']
    
    raw_dir = ingestion_params['raw_data_dir']
    processed_dir = ingestion_params['processed_data_dir']
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    # Separatori semantici per leggi (Articoli)
    custom_separators = [
        "\nArticle ", "\nArticolo ", 
        "\n\n", "\n", " ", ""
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=ingestion_params['chunk_size'],
        chunk_overlap=ingestion_params['chunk_overlap'],
        separators=custom_separators
    )
    
    all_docs = []
    
    # Processa AI Act e GDPR se presenti
    pdf_files = [f for f in os.listdir(raw_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"⚠ Nessun file PDF trovato in {raw_dir}. Assicurati di aver fatto 'dvc pull'.")
        return

    for pdf_file in pdf_files:
        print(f"Analizzando {pdf_file}...")
        pdf_path = os.path.join(raw_dir, pdf_file)
        raw_pages = extract_text_from_pdf(pdf_path)
        
        # Estraiamo il testo grezzo (potevamo farlo anche articolo per articolo)
        text = process_law_text(raw_pages, pdf_file)
        
        # Creiamo i chunk
        chunks = splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            all_docs.append({
                "source": pdf_file,
                "chunk_id": i,
                "content": chunk
            })
            
    # Salva il risultato
    output_path = os.path.join(processed_dir, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)
        
    print(f"✓ Ingestion completata. {len(all_docs)} chunk salvati in {output_path}")

if __name__ == "__main__":
    main()
