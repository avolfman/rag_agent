# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) system that allows you to ask questions about your documents using natural language.

## Features

- PDF document processing and chunking
- Vector-based semantic search using FAISS
- Natural language question answering
- REST API interface
- Automatic document processing and indexing
- Support for multiple documents

## Prerequisites

- Python 3.11 or higher
- CUDA 12.6 (optional, for GPU acceleration)
- Visual Studio Build Tools 2019 (for Windows)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a conda environment:
```bash
conda create -n llm_p11 python=3.11
conda activate llm_p11
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install FAISS (choose one based on your needs):
   - For CPU only:
   ```bash
   pip install faiss-cpu
   ```
   - For GPU support (requires CUDA 12.6):
   ```bash
   pip install faiss-gpu
   ```

## Configuration

1. Create a `config.yaml` file in the root directory:
```yaml
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
llm_model: "meta-llama/Llama-3.1-8B-Instruct"
```

2. Create a `data` directory in the root folder and place your PDF documents there.

## Usage

1. Start the API server:
```bash
python api_server.py
```

2. In a separate terminal, run the test client:
```bash
python test_client.py
```

The test client will:
- Process any new PDF files in the `data` directory
- Split them into chunks
- Add them to the RAG system
- Ask example questions about the documents

## API Endpoints

- `GET /health` - Check server health
- `POST /add_documents` - Add documents to the RAG system
- `POST /query` - Ask questions about the documents

## Troubleshooting

1. If you get timeout errors:
   - The server and client are configured with 3-minute timeouts
   - Check if your documents are very large
   - Ensure you have enough system resources

2. If FAISS installation fails:
   - For Windows: Make sure Visual Studio Build Tools 2019 is installed
   - Try the CPU version first: `pip install faiss-cpu`
   - For GPU support, ensure CUDA 12.6 is properly installed

3. If the server fails to start:
   - Check if port 8000 is available
   - Ensure all dependencies are installed
   - Check the logs for specific error messages

## Project Structure

```
.
├── api_server.py      # FastAPI server implementation
├── rag_agent.py       # RAG system core logic
├── test_client.py     # Example client implementation
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
└── data/             # Directory for PDF documents
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license] 