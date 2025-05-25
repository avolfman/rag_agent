import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any
from langchain.schema import Document
from rag_agent import RAGAgent
import atexit
from contextlib import asynccontextmanager
import logging
import signal
import sys
import time
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for RAG agent
rag_agent = None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def cleanup_database():
    """Safely remove the database directory"""
    try:
        if os.path.exists("faiss_index"):
            # Wait a bit for any operations to complete
            time.sleep(1)
            
            # Try to remove the directory
            shutil.rmtree("faiss_index", ignore_errors=True)
            logger.info("Database directory removed successfully")
    except Exception as e:
        logger.error(f"Error cleaning up database: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    global rag_agent
    try:
        # Startup
        logger.info("Initializing RAG Agent...")
        cleanup_database()  # Clean up any existing database
        rag_agent = RAGAgent()
        logger.info("RAG Agent initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Error initializing RAG Agent: {str(e)}")
        raise

app = FastAPI(lifespan=lifespan)

class DocumentRequest(BaseModel):
    documents: List[Dict[str, Any]]
    model_config = ConfigDict(arbitrary_types_allowed=True)

class Query(BaseModel):
    question: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

class Document(BaseModel):
    page_content: str
    metadata: Dict[str, Any]
    model_config = ConfigDict(arbitrary_types_allowed=True)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not rag_agent:
        raise HTTPException(status_code=503, detail="RAG agent not initialized")
    return {"status": "healthy"}

@app.post("/add_documents")
async def add_documents(documents: List[Document]):
    """Add documents to the RAG system."""
    if not rag_agent:
        raise HTTPException(status_code=503, detail="RAG agent not initialized")
        
    if not documents:
        raise HTTPException(status_code=400, detail="No documents provided")
        
    try:
        # Convert Pydantic documents to Langchain documents
        langchain_docs = [
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata
            ) for doc in documents
        ]
        
        # Add documents with retry logic
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                rag_agent.add_documents(langchain_docs)
                return {"status": "success", "message": f"Added {len(documents)} documents successfully"}
            except Exception as e:
                logger.error(f"Error adding documents (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error in add_documents endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(query: Query):
    """Query the RAG system."""
    if not rag_agent:
        raise HTTPException(status_code=503, detail="RAG agent not initialized")
        
    try:
        answer = rag_agent.query(query.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=120
    ) 