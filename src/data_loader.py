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