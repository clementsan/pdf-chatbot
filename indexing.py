"""
Indexing with vector database
"""

from pathlib import Path
import re

import chromadb

from unidecode import unidecode

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings



# Load PDF document and create doc splits
def load_doc(list_file_path, chunk_size, chunk_overlap):
    """Load PDF document and create doc splits"""

    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits


# Generate collection name for vector database
#  - Use filepath as input, ensuring unicode text
#  - Handle multiple languages (arabic, chinese)
def create_collection_name(filepath):
    """Create collection name for vector database"""

    # Extract filename without extension
    collection_name = Path(filepath).stem
    # Fix potential issues from naming convention
    ## Remove space
    collection_name = collection_name.replace(" ", "-")
    ## ASCII transliterations of Unicode text
    collection_name = unidecode(collection_name)
    ## Remove special characters
    collection_name = re.sub("[^A-Za-z0-9]+", "-", collection_name)
    ## Limit length to 50 characters
    collection_name = collection_name[:50]
    ## Minimum length of 3 characters
    if len(collection_name) < 3:
        collection_name = collection_name + "xyz"
    ## Enforce start and end as alphanumeric character
    if not collection_name[0].isalnum():
        collection_name = "A" + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + "Z"
    print("\n\nFilepath: ", filepath)
    print("Collection name: ", collection_name)
    return collection_name


# Create vector database
def create_db(splits, collection_name):
    """Create embeddings and vector database"""

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        # model_name="sentence-transformers/all-MiniLM-L6-v2",
        # model_kwargs={"device": "cpu"},
        # encode_kwargs={'normalize_embeddings': False}
    )
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name,
        # persist_directory=default_persist_directory
    )
    return vectordb
