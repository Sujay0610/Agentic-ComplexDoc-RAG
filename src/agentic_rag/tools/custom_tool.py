import os
import time # Added for delay

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

# Import necessary libraries for Chroma integration
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
# Commented out Pinecone imports
# from langchain_community.vectorstores import Pinecone as PineconeVectorstore
# from pinecone import Pinecone, ServerlessSpec

class DocumentSearchToolInput(BaseModel):
    query: str = Field(description="Query to search the document.")

class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    
    model_config = ConfigDict(extra="allow")
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.collection_name = "agentic-rag-collection" # Define your Chroma collection name
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = self._initialize_chroma_and_upsert_document()

    def _initialize_chroma_and_upsert_document(self):
        # No API key needed for Chroma - it's a local vector database
        print(f"Initializing Chroma collection: {self.collection_name}")
        
        # Load and split document
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create Chroma vectorstore and add documents
        # Chroma will create a local database automatically
        vectorstore = Chroma.from_documents(
            texts,
            self.embeddings,
            collection_name=self.collection_name,
            persist_directory="./chroma_db"  # Local directory to store the database
        )
        
        print(f"Chroma collection '{self.collection_name}' created successfully with {len(texts)} documents.")
        return vectorstore

    def _run(self, query: str) -> str:
        # Perform similarity search
        docs = self.vectorstore.similarity_search(query, k=10)
        
        # Format the results
        formatted_results = ""
        for doc in docs:
            formatted_results += f"{doc.page_content}\n____\n"
        
        return formatted_results.rstrip('____\n')
