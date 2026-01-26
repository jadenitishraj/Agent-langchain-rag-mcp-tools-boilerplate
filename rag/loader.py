import os
import pdfplumber
from typing import List

def load_pdf_text(path: str) -> str:
    """Load text from a single PDF file."""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def load_all_pdfs(folder_path: str) -> List[dict]:
    """Load all PDFs from a folder and return list of {filename, text}."""
    documents = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
        print("Please add your PDF files to this folder and run again.")
        return documents
    
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return documents
    
    for filename in pdf_files:
        filepath = os.path.join(folder_path, filename)
        print(f"  Loading: {filename}")
        text = load_pdf_text(filepath)
        documents.append({
            "filename": filename,
            "text": text
        })
    
    return documents
