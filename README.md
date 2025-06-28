# Agentic RAG over Complex Real-World Documents

This project demonstrates an enterprise-grade, agentic Retrieval-Augmented Generation (RAG) system designed to work with complex real-world documents. It leverages EyelevelAI's GroundX for advanced document parsing and retrieval, integrated as a custom tool with CrewAI.

## Features

- **Agentic RAG**: Utilizes CrewAI agents for intelligent document retrieval and response synthesis.
- **GroundX Integration**: Seamlessly integrates with GroundX for robust document ingestion and semantic search capabilities.
- **Streamlit UI**: Provides an interactive web interface for easy interaction and document querying.
- **Flexible Document Analysis**: Supports both simple PDF processing and complex document analysis (with images).

## Implementation Details

This project is built around the concept of agentic RAG, where specialized agents collaborate to answer user queries. Here's a breakdown of the key components:

- **CrewAI Agents**: The core of the system involves two main agents:
    - **Retriever Agent**: This agent is responsible for fetching relevant information. It is equipped with two tools:
        - `DocumentSearchTool`: A custom tool that interfaces with GroundX to perform semantic searches within uploaded PDF documents. It handles document ingestion, processing status polling, and content retrieval.
        - `SerperDevTool`: A tool for performing web searches, used as a fallback if relevant information is not found within the provided documents.
    - **Response Synthesizer Agent**: This agent takes the information retrieved by the Retriever Agent and synthesizes it into a concise and coherent answer to the user's query.

- **GroundX**: This powerful document processing and retrieval system is used to:
    - **Ingest Documents**: PDFs uploaded by the user are ingested into GroundX, which handles the complex task of parsing, extracting text (including from images), and preparing the content for search.
    - **Semantic Search**: GroundX provides advanced semantic search capabilities, allowing the `DocumentSearchTool` to find highly relevant passages based on the meaning of the query, rather than just keywords.

- **Streamlit Application (`app.py`)**: The Streamlit application provides the user interface. It allows users to:
    - Upload PDF documents.
    - Choose between simple PDF processing (using `SimpleDocumentSearchTool`) or complex document analysis (using `ComplexDocumentSearchTool`, which leverages GroundX's advanced features).
    - Ask questions related to the uploaded documents or general queries that might require web search.
    - View the conversation history and the agent's responses.

- **Custom Tools (`src/agentic_rag/tools/`)**: The `DocumentSearchTool` (both simple and complex versions) are custom implementations that wrap the GroundX API, making it accessible to the CrewAI agents. These tools manage the lifecycle of document processing and search queries with GroundX.

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

- Python 3.10 or later
- Access to GroundX API keys
- Access to Serper API keys (for web search functionality)

### 1. Setup Environment Variables

Create a `.env` file in the root directory of the project and add your API keys. You can refer to `.env.example` for the required variables.

```
GROUNDX_API_KEY="your_groundx_api_key_here"
SERPER_API_KEY="your_serper_api_key_here"
```

- **GroundX API keys**: Obtain your keys from [GroundX Documentation](https://docs.eyelevel.ai/documentation/fundamentals/quickstart#step-1-getting-your-api-key).
- **SERPER API keys**: Obtain your keys from [Serper.dev](https://serper.dev/).

### 2. Install Dependencies

Navigate to the project root directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Running the Application

Once the dependencies are installed, you can run the Streamlit application:

```bash
streamlit run app.py
```

This will open the application in your web browser, typically at `http://localhost:8501`.

## Dockerization

To containerize the application using Docker, follow these steps:

### 1. Build the Docker Image

Navigate to the project root directory (where the `Dockerfile` is located) and run the following command:

```bash
docker build -t agentic-rag-app .
```

This command builds a Docker image named `agentic-rag-app`.

### 2. Run the Docker Container

Once the image is built, you can run the application in a Docker container:

```bash
docker run -p 8501:8501 agentic-rag-app
```

This command maps port 8501 from your host to port 8501 in the container, allowing you to access the Streamlit application in your browser at `http://localhost:8501`.

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
