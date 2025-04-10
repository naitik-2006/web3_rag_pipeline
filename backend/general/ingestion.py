"""
Ingestion Pipeline Module

This module implements the ingestion pipeline for the RAG system using LangChain.
It:
  - Fetches documents from ArXiv using a query.
  - Optionally retrieves additional documents from TavilySearch.
  - Splits documents into smaller chunks.
  - Computes embeddings using a HuggingFace model.
  - Creates and persists a vector store using FAISS.
"""

import os
import logging
from typing import List, Dict, Any

from langchain_community.document_loaders import ArxivLoader
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import settings (API keys, model names, and vector store options)
from config.settings import (
    TAVILY_API_KEY,
    EMBEDDING_MODEL_NAME, 
    VECTORSTORE_SAVE_PATH
)

# Configure logger for detailed output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_arxiv_documents(query: str, max_results: int) -> List[Any]:
    """
    Fetch ArXiv documents for the given query using ArxivLoader.
    
    Args:
        query (str): The search query (e.g., 'bitcoin').
        max_results (int): Maximum number of results to fetch.
    
    Returns:
        List[Any]: A list of documents loaded from ArXiv.
    """
    logger.info("Fetching documents from ArXiv...")
    loader = ArxivLoader(query=query, max_results=max_results)
    documents = loader.load()
    logger.info(f"Fetched {len(documents)} documents from ArXiv.")
    return documents

def fetch_tavily_documents(query: str, k: int = 3) -> List[Any]:
    """
    Fetch documents using TavilySearch retriever (using LangChain's custom retriever).
    
    Args:
        query (str): The search query.
        k (int): Number of relevant documents to retrieve.
    
    Returns:
        List[Any]: A list of documents retrieved from TavilySearch.
    """
    logger.info("Fetching documents using TavilySearch retriever...")
    # Initialize the custom TavilySearch retriever. The API key is assumed to be configured.
    tavily_retriever = TavilySearchAPIRetriever(k=k)
    documents = tavily_retriever.invoke(query)
    print(documents)
    logger.info(f"Fetched {len(documents)} documents using TavilySearch.")
    return documents

def split_documents(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    """
    Split the documents into smaller chunks suitable for vectorization.
    
    Args:
        documents (List[Any]): A list of documents.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.
    
    Returns:
        List[Any]: The list of document chunks.
    """
    logger.info("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(documents)
    logger.info(f"Split documents into {len(split_docs)} chunks.")
    return split_docs

def create_and_save_vectorstore(
    documents: List[Any],
    embedding_model_name: str,
    save_path: str,
) -> None:
    """
    Create a vector store from the documents using the specified embedding model and vectorstore type.
    The vector store is persisted locally.
    
    Args:
        documents (List[Any]): List of document chunks.
        embedding_model_name (str): The HuggingFace embedding model name.
        save_path (str): Directory where the vector store will be saved.
    """
    logger.info("Initializing the embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(os.path.join(save_path, "faiss_index"))
   
    logger.info("Vectorstore created and saved successfully.")
    return vectorstore

def ingest_pipeline(arxiv_query = None , tavily_query = None) -> None:
    """
    Orchestrates the ingestion process:
      - Fetch documents from multiple sources.
      - Combine and split them.
      - Compute embeddings and index the documents in the vector store.
    """
    # Fetch documents from ArXiv and TavilySearch
    all_docs = []
    tavily_docs = None
    vectorstore = None
    if arxiv_query : 
        arxiv_docs = fetch_arxiv_documents(arxiv_query, 2)
        all_docs = all_docs + arxiv_docs
        
        # Split documents into manageable chunks
        split_docs = split_documents(all_docs)
        # Create and save vector store locally
        vectorstore = create_and_save_vectorstore(split_docs, EMBEDDING_MODEL_NAME, VECTORSTORE_SAVE_PATH)
    if tavily_query : 
        tavily_docs = fetch_tavily_documents(tavily_query)
        # Combine the documents from both sources
        all_docs = all_docs + tavily_docs
    
    logger.info(f"Total documents fetched: {len(all_docs)}")
    logger.info("Ingestion pipeline completed successfully.")
    
    return tavily_docs, vectorstore


# Module entry point for testing or standalone run
if __name__ == "__main__":
    ingest_pipeline(arxiv_query=None , tavily_query="Tell me more the idea of bitcoin")
