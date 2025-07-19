# main.py

import os
from dotenv import load_dotenv
from src.data_loader import load_all_pdfs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Importing the new function and other necessary components for Full Adjudication Workflow
from src.llm_handler import get_structuring_chain, get_adjudication_chain
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
import json

#  importing function to parese the input query
from src.data_loader import extract_text_from_file
from src.llm_handler import get_structuring_chain, get_adjudication_chain
from langchain_community.document_loaders import UnstructuredEmailLoader


import nltk
# Download the 'punkt' toolkit from NLTK if it's missing
nltk.download('punkt')



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

    # --- 4. Test the Query Structuring Chain ---
    print("\n--- Testing the Query Structuring Chain ---")
    # Get the chain from our handler
    structuring_chain = get_structuring_chain(api_key=token, base_url=endpoint)
    adjudication_chain = get_adjudication_chain(api_key=token, base_url=endpoint)
    retriever = vectorstore.as_retriever()

    full_chain = {
        "claim_details": structuring_chain,
        "context": itemgetter("query") | retriever
    } | adjudication_chain
    
    # --- 5. Run the Chain on an Input File ---
    # Define the path to your input query file (e.g., an email saved as a PDF or Word doc)
    input_file_path = "queries/sample8HTML.html" # <-- THIS IS THE PATH TO THE INPUT FILE
    
    # Check if the file exists before proceeding
    if not os.path.exists(input_file_path):
        print(f"❌ Error: Input file not found at '{input_file_path}'")
        return
        
    # Extract text from the file to use as the query
    raw_query = extract_text_from_file(input_file_path)
    
    print("\n--- Running Full Adjudication Chain ---")
    print(f"Query (from file): {raw_query[:500]}...\n") # Print first 500 chars

    # Invoke the full chain with the extracted text
    final_result = full_chain.invoke({"query": raw_query})

    # Print the final structured output
    print("--- Final Decision ---")
    print(json.dumps(final_result, indent=2))


if __name__ == "__main__":
    main()