"""
FAISS Index Creation Module

This module loads email data from a JSONL file (created from the Bitcoindev mailing list),
converts each entry into a Document, creates a FAISS index with HuggingFace embeddings,
and saves the index locally. If a saved FAISS index already exists, the module loads
and returns it.

Functions:
    load_emails(file_path: str) -> List[Document]:
        Reads the JSONL file and converts each line into a Document.
        
    create_faiss_index(file_path: str, save_dir: str) -> Tuple[FAISS, List[Document]]:
        Creates or loads the FAISS index from the given file path and save directory.

Usage:
    Run this file as a script to build and save the FAISS index.
    The function returns both the vectorstore and the documents.
"""

import os
import json
import logging
from typing import List, Tuple, Optional
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import STATIC_VS_SAVE_PATH, EMBEDDING_MODEL_NAME

# Configure logging for the module.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_emails(file_path: str) -> List[Document]:
    """
    Load email data from a JSONL file and convert each entry into a Document.
    
    Each line in the JSONL file should be a JSON object with at least the following fields:
        - commit
        - subject
        - author
        - date
        - body
        
    The Document's page_content is created by concatenating the subject and body,
    and metadata is stored for further retrieval.
    
    Args:
        file_path (str): Path to the JSONL file containing email data.
    
    Returns:
        List[Document]: A list of Document objects.
    """
    documents = []
    logger.info(f"Loading emails from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            email = json.loads(line.strip())
            # Create text combining subject and body.
            text = f"{email.get('subject', '')}\n{email.get('body', '')}"
            metadata = {
                "commit": email.get("commit", ""),
                "subject": email.get("subject", ""),
                "author": email.get("author", ""),
                "date": email.get("date", ""),
            }
            documents.append(Document(page_content=text, metadata=metadata))
    logger.info(f"Loaded {len(documents)} documents from the file.")
    return documents

def create_faiss_index(file_path: str, save_dir: str = STATIC_VS_SAVE_PATH) -> Tuple[FAISS, Optional[List[Document]]]:
    """
    Creates a FAISS index from email documents loaded from a JSONL file.
    
    If a saved FAISS index already exists at the specified path, it is loaded and returned.
    Otherwise, the index is built from scratch, saved, and the list of documents is returned.
    
    Args:
        file_path (str): Path to the JSONL file containing email data.
        save_dir (str): Directory where the FAISS index should be saved/loaded.
    
    Returns:
        Tuple[FAISS, Optional[List[Document]]]:
            - The FAISS vectorstore.
            - The list of Documents if the index was newly created, otherwise None.
    """
    index_path = os.path.join(save_dir.split(".")[0], "faiss_index")
    logger.info(f"Looking for FAISS index at {index_path}...")
    
    # Initialize the embedding model.
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    documents = load_emails(file_path)
    
    if os.path.exists(index_path):
        logger.info("Existing FAISS index found. Loading the index...")
        vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization = True)
        return vectorstore, documents
    else:
        logger.info("No FAISS index found. Creating a new index...")
        # Load emails and convert to documents.
        vectorstore = FAISS.from_documents(documents, embedding_model)
        logger.info("Saving new FAISS index...")
        vectorstore.save_local(index_path)
        return vectorstore, documents


if __name__ == "__main__":
    # Assuming the emails file is located in the project root.
    EMAILS_FILE = os.path.join(os.path.dirname(__file__), "..", "emails_bitcoindev.jsonl")
    
    # Create or load the FAISS index.
    vectorstore, docs = create_faiss_index(EMAILS_FILE)