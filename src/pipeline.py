# In src/pipeline.py

import os
import json
import nltk
from dotenv import load_dotenv
from operator import itemgetter
from src.data_loader import load_all_pdfs, extract_text_from_file
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from src.llm_handler import get_structuring_chain, get_adjudication_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def run_pipeline(input_file_path):
    """
    Runs the full adjudication pipeline on a given input file.
    """
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    endpoint = "https://models.github.ai/inference"
    persist_directory = "data/vector_store_db"

    if not token:
        return {"error": "API token not found."}

    embedding_model = OpenAIEmbeddings(
        api_key=token, base_url=endpoint, model="text-embedding-3-small", chunk_size=32
    )
    llm = ChatOpenAI(model="openai/gpt-4o", api_key=token, base_url=endpoint, temperature=0.0)

    vectorstore = Chroma(
        persist_directory=persist_directory, embedding_function=embedding_model
    )
    
    base_retriever = vectorstore.as_retriever()
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    structuring_chain = get_structuring_chain(api_key=token, base_url=endpoint)
    adjudication_chain = get_adjudication_chain(api_key=token, base_url=endpoint)

    full_chain = {
        "claim_details": structuring_chain,
        "context": itemgetter("query") | compression_retriever
    } | adjudication_chain

    if not os.path.exists(input_file_path):
        return {"error": f"Input file not found at '{input_file_path}'"}
        
    raw_query = extract_text_from_file(input_file_path)
    
    final_result = full_chain.invoke({"query": raw_query})
    
    return final_result