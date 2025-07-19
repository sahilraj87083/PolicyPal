# main.py

import os
from dotenv import load_dotenv
from src.data_loader import load_all_pdfs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def main():
    """
    Main function to run the PolicyPal application.
    It loads data, creates or loads a vector store, and prepares for querying.
    """
    # --- 1. Configuration and Setup ---
    # Load credentials from the .env file
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    endpoint = "https://models.github.ai/inference"

    # Define the directory to save the persistent vector store
    persist_directory = "data/vector_store_db"

    # Check for the API token
    if not token:
        print("❌ GITHUB_TOKEN not found. Make sure it's in your .env file.")
        return # Exit the function if the token is not found

    print("--- Starting PolicyPal ---")

    # Initialize the embedding model, which is needed for both creating and loading the vector store
    embedding_model = OpenAIEmbeddings(
        api_key=token,
        base_url=endpoint,
        model="text-embedding-3-small",
        chunk_size=32
    )

    # --- 2. Vector Store Management ---
    # Check if the vector store already exists on disk
    if os.path.exists(persist_directory):
        # If it exists, load it directly from the disk
        print(f"✅ Loading existing vector store from '{persist_directory}'...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        print("✅ Vector store loaded successfully!")
    else:
        # If it does not exist, create it from scratch
        print(f"--- No existing vector store found. Creating a new one in '{persist_directory}' ---")
        
        # Step 2a: Load PDF documents from the 'data' folder
        all_policy_docs = load_all_pdfs(data_folder_path="data")
        print(f"📄 Total pages loaded: {len(all_policy_docs)}")

        # Step 2b: Split the loaded documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_policy_docs)
        print(f"Total chunks created: {len(splits)}")

        # Step 2c: Create the vector store from the chunks and save it to disk
        print("\nCreating and persisting new vector store... (This may take a few moments)")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        print("✅ New vector store created and saved successfully!")

    # --- 3. Application Logic ---
    # Now that 'vectorstore' is ready (either loaded or newly created),
    # you can build the rest of your application here.
    # For example, you can start asking questions.
    print("\n--- PolicyPal is ready to answer questions ---")
    # (The query logic will go here in the next steps)


if __name__ == "__main__":
    main()