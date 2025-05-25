import requests
import os
import json
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server configuration
SERVER_URL = "http://localhost:8000"
DATA_FOLDER = "data"
PROCESSED_FILES = set()

# Configure requests session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

def read_pdf(file_path):
    """Read and extract text from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {str(e)}")
        return None

def add_documents(documents):
    """Add documents to the RAG system."""
    try:
        response = requests.post(
            f"{SERVER_URL}/add_documents",
            json=documents,  # Send documents directly without wrapping in a dict
            timeout=180  # Increase timeout to 3 minutes
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error adding documents: {str(e)}")
        raise

def ask_question(question):
    """Ask a question to the RAG system."""
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json={"question": question},
            timeout=180  # Increase timeout to 3 minutes
        )
        response.raise_for_status()
        return response.json()["answer"]
    except Exception as e:
        logger.error(f"Error asking question: {str(e)}")
        raise

def process_data_folder():
    """Process all PDF files in the data folder."""
    if not os.path.exists(DATA_FOLDER):
        logger.error(f"Data folder {DATA_FOLDER} does not exist")
        return None
        
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Match the server's chunk size
        chunk_overlap=50,  # Match the server's overlap
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    new_files = []
    all_chunks = []
    
    # Process each PDF file
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith('.pdf') and filename not in PROCESSED_FILES:
            file_path = os.path.join(DATA_FOLDER, filename)
            logger.info(f"Processing file: {filename}")
            
            # Read PDF
            text = read_pdf(file_path)
            if not text:
                continue
                
            # Split text into chunks
            chunks = text_splitter.split_text(text)
            logger.info(f"Created {len(chunks)} chunks from {filename}")
            
            # Format chunks as documents
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "page_content": chunk,
                    "metadata": {
                        "source": filename,
                        "chunk": i + 1,
                        "total_chunks": len(chunks)
                    }
                })
            
            new_files.append(filename)
            PROCESSED_FILES.add(filename)
    
    if new_files:
        logger.info(f"Found {len(new_files)} new files: {', '.join(new_files)}")
        logger.info(f"Created {len(all_chunks)} total chunks")
        return all_chunks
    else:
        logger.info("No new files found")
        return None

def check_server_status():
    """Check if the server is running and healthy."""
    try:
        response = session.get(f"{SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def main():
    """Main function to process documents and handle queries."""
    # Check server status
    if not check_server_status():
        logger.error("Server is not running. Please start the server first.")
        return
        
    # Process new documents
    chunks = process_data_folder()
    if chunks:
        logger.info(f"Adding {len(chunks)} chunks to RAG system...")
        if add_documents(chunks):
            logger.info("Documents added successfully")
        else:
            logger.error("Failed to add documents")
            return
    
    # Example questions
    questions = [
        "What is the camera's frequency and sampling rate?",
        "What are the technical specifications of the camera?",
        "What is the resolution of the camera?"
    ]
    
    # Ask questions
    for question in questions:
        logger.info(f"\nAsking: {question}")
        answer = ask_question(question)
        if answer:
            logger.info(f"Answer: {answer}")
        else:
            logger.error("Failed to get answer")

if __name__ == "__main__":
    main() 