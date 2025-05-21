import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self, model_name="microsoft/phi-2"):
        self.model_name = model_name
        self.vector_store = None
        self.qa_chain = None
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize the embeddings and language model components."""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            
            # Initialize language model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map="auto"
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                num_return_sequences=1,
                return_full_text=False
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("Successfully initialized embeddings and language model")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def add_documents(self, documents):
        """Add documents to the vector store."""
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return
                
            # Create text splitter with smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                length_function=len,
            )
            
            # Split documents into chunks
            texts = text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(texts)} chunks")
            
            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(texts, self.embeddings)
                logger.info("Created new FAISS vector store")
            else:
                self.vector_store.add_documents(texts)
                logger.info("Updated existing FAISS vector store")
            
            # Create QA chain with optimized prompt
            prompt_template = """Answer the question based on the context provided. Keep the answer concise and to the point.

Context: {context}

Question: {question}
Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 2}),
                chain_type_kwargs={"prompt": PROMPT}
            )
            logger.info("Created new QA chain")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def query(self, question):
        """Query the RAG system."""
        try:
            if not self.qa_chain:
                return "Please add some documents first using add_documents()"
            
            result = self.qa_chain.invoke({"query": question})
            return result["result"].strip()
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            return f"Error processing query: {str(e)}"
    
    def save(self, directory="faiss_index"):
        """Save the vector store to disk."""
        try:
            if self.vector_store:
                self.vector_store.save_local(directory)
                logger.info(f"Saved vector store to {directory}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load(self, directory="faiss_index"):
        """Load the vector store from disk."""
        try:
            self.vector_store = FAISS.load_local(directory, self.embeddings)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 2})
            )
            logger.info(f"Loaded vector store from {directory}")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

def main():
    # Example usage
    agent = RAGAgent()
    
    # Add some example documents
    documents = [
        {
            "page_content": "The quick brown fox jumps over the lazy dog.",
            "metadata": {"source": "example1"}
        },
        {
            "page_content": "Python is a high-level programming language.",
            "metadata": {"source": "example2"}
        }
    ]
    
    agent.add_documents(documents)
    
    # Ask a question
    question = "What is Python?"
    answer = agent.query(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main() 