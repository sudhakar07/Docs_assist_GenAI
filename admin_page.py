import os
import streamlit as st
import google.generativeai as genai
import faiss
import pickle
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class VectorDBManager:
    def __init__(self, db_dir='vector_databases'):
        self.db_dir = db_dir
        os.makedirs(db_dir, exist_ok=True)
        
        # Configure Gemini API
        
    
    def generate_embeddings(self, texts):
        """
        Generate embeddings using Gemini Pro API
        
        Args:
            texts (List[str]): List of text chunks to embed
        
        Returns:
            List[List[float]]: List of embedding vectors
        """
        model="models/text-embedding-004"
        embeddings = []
        
        for text in texts:
            try:
                # Use embed_content method
                result = genai.embed_content(
                    model=model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                st.error(f"Embedding error: {e}")
                # Append a zero vector if embedding fails
                embeddings.append([0.0] * 768)  # Assuming 768-dimensional embedding
        
        return embeddings
    
    def create_vector_db(self, pdf_files, db_name):
        # Validate inputs
        if not pdf_files or not db_name:
            st.error("Please provide PDF files and a database name")
            return None
        
        # Text extraction and chunking
        text_chunks = []
        metadata = []
                        # Advanced text splitting
        text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
        for pdf in pdf_files:
            try:
                with pdf as f:
                    pdf_reader = PdfReader(f)
                    # Extract text from all pages
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()
                    
                    # st.write(pdf_text)
                    # Split text into chunks
                    chunks = text_splitter.split_text(pdf_text)
                    text_chunks.extend(chunks)
                    
                    # Create metadata for each chunk
                    file_metadata = {
                        'filename': pdf.name,
                        'total_pages': len(pdf_reader.pages)
                    }
                    metadata.extend([file_metadata] * len(chunks))
            
            except Exception as e:
                st.error(f"Error processing PDF {pdf.name}: {e}")
                continue
        
        # Generate embeddings
        try:
            embeddings = self.generate_embeddings(text_chunks)

            #st.write(embeddings)
        except Exception as e:
            st.error(f"Embedding generation failed: {e}")
            return None
        
        # Create FAISS index
        try:
            db_path = os.path.join(self.db_dir, f"{db_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            embedding_dim = len(embeddings[0])  # Get the dimensionality of the embeddings
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(np.array(embeddings).astype(np.float32))
            index_path = f"{db_path}_index.faiss"
            faiss.write_index(index, index_path)
            st.success("Embedding and index created and saved successfully.")
        except Exception as e:
            st.error(f"FAISS index creation failed: {e}")
            return None
        
        # Save knowledge base
        try:
            db_path = os.path.join(self.db_dir, f"{db_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            
            
            with open(f"{db_path}_metadata.pkl", 'wb') as f:
                pickle.dump({
                    'chunks': text_chunks,
                    'metadata': metadata,
                    'embeddings': embeddings
                }, f)
            
            st.success(f"knowledge base '{db_name}' created successfully!")
            return db_path
        
        except Exception as e:
            st.error(f"Database saving failed: {e}")
            return None

    # ... (rest of the methods remain the same as in previous implementation)

def admin_page():
    st.title("📋 Admin Operations")
    
    # Hardcoded login
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if not st.session_state['logged_in']:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username == "admin" and password == "admin123":
                st.session_state['logged_in'] = True
                #st.experimental_rerun()
            else:
                st.error("Invalid Credentials")
    
    if st.session_state['logged_in']:
        vector_db_manager = VectorDBManager()
        
        # Logout button
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            #st.experimental_rerun()
        
        # knowledge base Operations
        st.header("knowledge base Management")
        
        # Create New knowledge base Section
        st.subheader("Create New knowledge base")
        new_pdf_files = st.file_uploader("Upload PDF Files for New Database", 
                                         type=['pdf'], 
                                         accept_multiple_files=True,
                                         key="new_db_upload")
        
        new_db_name = st.text_input("New knowledge base Name")
        
        if st.button("Create New knowledge base") and new_pdf_files and new_db_name:
            vector_db_manager.create_vector_db(new_pdf_files, new_db_name)
        
        # ... (rest of the method remains the same)

if __name__ == "__main__":
    
    admin_page()