import os
import time
from crewai.tools import BaseTool
from typing import Type, Optional, List, Dict
from pydantic import BaseModel, Field, ConfigDict
from groundx import Document, GroundX
from dotenv import load_dotenv

load_dotenv()


class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the document.")

class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    
    model_config = ConfigDict(extra="allow")
    
    def __init__(self, file_path: Optional[str] = None, bucket_id: Optional[str] = None):
        """Initialize the searcher with either a new PDF file or existing bucket ID."""
        super().__init__()
        self.file_path = file_path
        self.client = GroundX(
            api_key=os.getenv("GROUNDX_API_KEY")
        )
        
        if bucket_id:
            # Use existing bucket
            self.bucket_id = bucket_id
            print(f"Using existing GroundX bucket: {bucket_id}")
        elif file_path:
            # Create new bucket and upload document
            self.bucket_id = self._create_bucket()
            self.process_id = self._upload_document()
            # Wait for document processing to complete during initialization
            self._wait_for_processing_completion()
        else:
            raise ValueError("Either file_path or bucket_id must be provided")
    
    def _upload_document(self):
        ingest = self.client.ingest(
                        documents=[
                            Document(
                            bucket_id=self.bucket_id,
                            file_name=os.path.basename(self.file_path),
                            file_path=self.file_path,
                            file_type="pdf",
                            search_data=dict(
                                key = "value",
                            ),
                            )
                        ]
                        )
        return ingest.ingest.process_id
    
    def _create_bucket(self):
        # Create bucket with timestamp to make it unique
        bucket_name = f"agentic_rag_{int(time.time())}"
        response = self.client.buckets.create(
            name=bucket_name
        )
        return response.bucket.bucket_id
    
    @staticmethod
    def list_existing_buckets() -> List[Dict]:
        """List all existing GroundX buckets with their documents."""
        client = GroundX(api_key=os.getenv("GROUNDX_API_KEY"))
        try:
            buckets_response = client.buckets.list()
            buckets_info = []
            
            for bucket in buckets_response.buckets:
                # Get documents in this bucket
                try:
                    # Use the correct API method without bucket_id parameter
                    docs_response = client.documents.list()
                    documents = []
                    
                    # Filter documents by bucket_id manually
                    for doc in docs_response.documents:
                        if hasattr(doc, 'bucket_id') and doc.bucket_id == bucket.bucket_id:
                            doc_status = getattr(doc, 'status', 'unknown')
                            documents.append({
                                "name": getattr(doc, 'file_name', 'Unknown'),
                                "status": doc_status,
                                "created_at": getattr(doc, 'created_at', 'Unknown'),
                                "document_id": getattr(doc, 'document_id', 'Unknown')
                            })
                    
                    buckets_info.append({
                        "bucket_id": bucket.bucket_id,
                        "bucket_name": getattr(bucket, 'name', f'Bucket_{bucket.bucket_id}'),
                        "created_at": getattr(bucket, 'created_at', 'Unknown'),
                        "documents": documents
                    })
                except Exception as e:
                    print(f"Error getting documents for bucket {bucket.bucket_id}: {e}")
                    buckets_info.append({
                        "bucket_id": bucket.bucket_id,
                        "bucket_name": getattr(bucket, 'name', f'Bucket_{bucket.bucket_id}'),
                        "created_at": getattr(bucket, 'created_at', 'Unknown'),
                        "documents": [],
                        "error": str(e)
                    })
            
            return buckets_info
        except Exception as e:
            print(f"Error listing buckets: {e}")
            return []    

    def _wait_for_processing_completion(self):
        """Wait for document processing to complete during initialization."""
        print("Waiting for GroundX document processing to complete...")
        timeout = 300  # 5 minutes timeout
        start_time = time.time()
        while True:
            try:
                status_response = self.client.documents.get_processing_status_by_id(
                    process_id=self.process_id
                )
                if status_response.ingest.status == "complete":
                    print("GroundX document processing complete! Ready for queries.")
                    break
                elif status_response.ingest.status == "error":
                    raise Exception(f"Document processing failed: {status_response.ingest.status}")
                elif time.time() - start_time > timeout:
                    raise Exception("Document processing timed out after 5 minutes.")
                else:
                    print(f"Document processing status: {status_response.ingest.status}... Still waiting...")
                    time.sleep(10)  # Wait for 10 seconds before checking again
            except Exception as e:
                print(f"Error checking processing status: {e}")
                time.sleep(5)  # Wait a bit before retrying
                if time.time() - start_time > timeout:
                    raise Exception("Document processing timed out.")
    
    def _run(self, query: str) -> str:
        """Search the document for the given query. Document should already be processed."""
        try:
            # Document should already be processed, so proceed directly with search
            search_response = self.client.search.content(
                id=self.bucket_id,
                query=query,
                n=10,
                verbosity=2
            )
            
            # Format the results with separators
            formatted_results = ""
            for result in search_response.search.results:
                formatted_results += f"{result.text}\n____\n"
            
            return formatted_results.rstrip('____\n')
        except Exception as e:
            return f"Error during search: {str(e)}"

# Test the implementation
def test_document_searcher():
    # Test file path
    pdf_path = "knowledge/demo.pdf"
    
    # Create instance
    searcher = DocumentSearchTool(file_path=pdf_path)
    
    # Test search
    result = searcher._run("What is the purpose of campedUI?")
    print("Search Results:", result)

if __name__ == "__main__":
    test_document_searcher()
