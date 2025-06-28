import os
import time # Added for delay

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

# Import necessary libraries for Pinecone integration
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorstore
from pinecone import Pinecone, ServerlessSpec

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
        self.index_name = "agentic-rag-index" # Define your Pinecone index name
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = self._initialize_pinecone_and_upsert_document()

    def _initialize_pinecone_and_upsert_document(self):
        api_key = os.getenv("PINECONE_API_KEY")
        

        if not api_key:
            raise ValueError("PINECONE_API_KEY must be set as environment variables.")

        pc = Pinecone(api_key=api_key)

        print(f"Attempting to create/connect to index: {self.index_name}")
        print(f"Existing indexes in Pinecone: {pc.list_indexes()}")

        # Check if index exists, if not, create it
        # Always ensure a fresh index for each use
        if self.index_name in pc.list_indexes():
            print(f"Index '{self.index_name}' already exists. Deleting it for a fresh start...")
            pc.delete_index(self.index_name)
            print(f"Index '{self.index_name}' deleted. Waiting for deletion to propagate...")
            time.sleep(10) # Wait for 10 seconds to ensure deletion propagates

        print(f"Creating new index: {self.index_name}...")
        pc.create_index(
            name=self.index_name,
            dimension=self.embeddings.client.get_sentence_embedding_dimension(), # Dimension of 'all-MiniLM-L6-v2'
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"Index '{self.index_name}' created successfully.")
        

        # Load and split document
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create Pinecone vectorstore and upsert documents
        vectorstore = PineconeVectorStore.from_documents(
            texts,
            self.embeddings,
            index_name=self.index_name
        )
        return vectorstore

    def _run(self, query: str) -> str:
        # Perform similarity search
        docs = self.vectorstore.similarity_search(query, k=10)
        
        # Format the results
        formatted_results = ""
        for doc in docs:
            formatted_results += f"{doc.page_content}\n____\n"
        
        return formatted_results.rstrip('____\n')
