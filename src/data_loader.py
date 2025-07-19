# In src/data_loader.py

import os
from langchain_community.document_loaders import PyPDFLoader

def load_all_pdfs(data_folder_path: str):
    """
    Loads all PDF documents from a specified folder.

    Args:
        data_folder_path: The path to the folder containing PDF files.

    Returns:
        A list of loaded document objects.
    """
    documents = []
    print(f"Loading all PDF documents from: {data_folder_path}")

    for filename in os.listdir(data_folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_folder_path, filename)
            try:
                loader = PyPDFLoader(file_path)
                # .load() returns a list of pages, so we extend our main list
                documents.extend(loader.load())
                print(f"  ✔ Successfully loaded '{filename}'")
            except Exception as e:
                print(f"  ❌ Failed to load '{filename}': {e}")
    
    if not documents:
        print("Warning: No documents were loaded. Check the folder path and PDF files.")
    
    return documents


# for reading the input files

from langchain_community.document_loaders import Docx2txtLoader

def extract_text_from_file(file_path: str):
    """
    Extracts text content from a given file (PDF or DOCX).
    
    Args:
        file_path: The path to the file.

    Returns:
        The extracted text content as a single string.
    """
    print(f"\n--- Extracting text from query file: {os.path.basename(file_path)} ---")
    
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        # For simplicity, we'll raise an error for unsupported types.
        # You could add email (.eml) or other parsers here.
        raise ValueError(f"Unsupported file type: {file_path}")

    # Load the document (it returns a list of pages/documents)
    docs = loader.load()
    
    # Join the content of all pages into a single string
    return "\n".join(doc.page_content for doc in docs)