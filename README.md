# PDF Document Q&A System

A RAG (Retrieval-Augmented Generation) based system for answering questions about PDF documents using LangChain, FAISS, and HuggingFace models.

## Features

- PDF document processing and text extraction
- Document chunking and vector storage using FAISS
- Question answering using HuggingFace models
- REST API server for document management and queries
- Client application for easy interaction

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-qa-system.git
cd pdf-qa-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your PDF documents in the `data` folder

2. Start the API server:
```bash
python api_server.py
```

3. Run the client to process documents and ask questions:
```bash
python test_client.py
```

## Project Structure

- `api_server.py`: FastAPI server for document management and queries
- `rag_agent.py`: RAG implementation using LangChain and FAISS
- `test_client.py`: Client application for document processing and queries
- `requirements.txt`: Project dependencies
- `data/`: Directory for PDF documents

## API Endpoints

- `GET /health`: Health check endpoint
- `POST /add_documents`: Add documents to the system
- `POST /query`: Query the system with questions

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 