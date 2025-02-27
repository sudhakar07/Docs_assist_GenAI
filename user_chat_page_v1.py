import os
import sqlite3
import streamlit as st
import google.generativeai as genai
import faiss
import pickle
import numpy as np
import uuid
import platform
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key_secrectpass = st.secrets["api_key"]
admin_secrectpass = st.secrets["admin_pass"]
genai.configure(api_key=api_key_secrectpass)

class UserActivityTracker:
    def __init__(self, db_path='user_activity.sqlite'):
        """
        Initialize SQLite database for user activity tracking
        
        Columns:
        - id: Unique identifier
        - timestamp: Datetime of interaction
        - username: User identifier
        - session_id: Unique session identifier
        - query: User's question
        - response: Model's response
        - feedback_type: Like/Dislike
        - feedback_reason: Optional user feedback reason
        - model_name: Model used
        - response_time: Time taken to generate response
        - source_document: Source document for response
        """
        self.db_path = db_path
        self._create_table()
        
    
    def _create_table(self):
        """Create SQLite table for user activity tracking"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    username TEXT,
                    session_id TEXT,
                    query TEXT,
                    response TEXT,
                    feedback_type TEXT,
                    feedback_reason TEXT,
                    model_name TEXT,
                    response_time REAL,
                    source_document TEXT
                )
            ''')
            conn.commit()
    
    def log_interaction(
        self, 
        username: str, 
        query: str, 
        response: str, 
        model_name: str,
        response_time: float,
        source_document: str = None,
        session_id: str = None
    ):
        """Log user interaction in database"""
        interaction_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_interactions (
                    id, timestamp, username, session_id, 
                    query, response, model_name, 
                    response_time, source_document
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                interaction_id, 
                datetime.now(), 
                username, 
                session_id, 
                query, 
                response, 
                model_name, 
                response_time, 
                source_document
            ))
            conn.commit()
        
        return interaction_id
    
    def log_feedback(
        self, 
        interaction_id: str, 
        feedback_type: str, 
        feedback_reason: str = None
    ):
        # st.write("interaction_id")
        # st.write(interaction_id)
        """Update interaction with user feedback"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE user_interactions 
                SET feedback_type = ?, feedback_reason = ? 
                WHERE id = ?
            ''', (feedback_type, feedback_reason, interaction_id))
            conn.commit()
    
    def get_analytics(self):
        """Generate analytics from user interactions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total interactions
            cursor.execute("SELECT COUNT(*) FROM user_interactions")
            total_interactions = cursor.fetchone()[0]
            
            # Feedback distribution
            cursor.execute("""
                SELECT 
                    feedback_type, 
                    COUNT(*) as count, 
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM user_interactions), 2) as percentage
                FROM user_interactions
                WHERE feedback_type IS NOT NULL
                GROUP BY feedback_type
            """)
            feedback_stats = cursor.fetchall()
            # st.write(feedback_stats)
            # Average response time
            cursor.execute("SELECT AVG(response_time) FROM user_interactions")
            avg_response_time = cursor.fetchone()[0]
            
            return {
                'total_interactions': total_interactions,
                'feedback_stats': feedback_stats,
                'avg_response_time': avg_response_time
            }

class EnhancedDocumentAssistant:
    def __init__(self, db_path):
        self.db_path = db_path
        self.tracker = UserActivityTracker()
        
        # Load FAISS index and metadata (same as before)
        self.index = faiss.read_index(db_path)
        
        
        with open(db_path.replace('_index.faiss', '_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.chunks = metadata['chunks']
        self.metadata = metadata['metadata']
        self.embeddings = metadata['embeddings']

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        # Generate query embedding
        # query_embedding = genai.embed_content(
        #     model="models/text-embedding-004",
        #     content=[query]
        # )[0].embedding

        result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=query,
                    task_type="retrieval_document"
                )
                
        query_embedding=result['embedding']

        # st.write(result['embedding'])
        # Search in FAISS index
        D, I = self.index.search(np.array([query_embedding], dtype='float32'), top_k)
        
        # Retrieve relevant chunks and their metadata
        relevant_results = [
            {
                'chunk': self.chunks[idx],
                'metadata': self.metadata[idx],
                'similarity_score': 1 / (1 + D[0][i])  # Convert distance to similarity
            } 
            for i, idx in enumerate(I[0])
        ]
        
        return relevant_results
    
    def generate_response(self, query: str, username: str) -> Dict:
        """Enhanced response generation with tracking"""
        start_time = datetime.now()
        
        # Retrieve relevant chunks (same retrieval logic)
        relevant_chunks = self.retrieve_relevant_chunks(query)
        context = "\n".join([chunk['chunk'] for chunk in relevant_chunks])
        
        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(
            f"Context: {context}\n\nQuery: {query}\n\nGenerate a comprehensive answer based on the context."
        )
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response with source metadata
        metadata_str = "\n\nRelevant Document Sources:"
        for chunk in relevant_chunks:
            metadata_str += f"\n- File: {chunk['metadata']['filename']} (Similarity: {chunk['similarity_score']:.2%})"
        
        full_response = response.text + metadata_str
        
        # Log interaction
        interaction_id = self.tracker.log_interaction(
            username=username,
            query=query,
            response=full_response,
            model_name='gemini-1.5-pro',
            response_time=response_time,
            source_document=', '.join(set(chunk['metadata']['filename'] for chunk in relevant_chunks))
        )
        
        return {
            'response': full_response,
            'interaction_id': interaction_id
        }
    
    # Existing methods like retrieve_relevant_chunks remain the same

def get_username():
    """Get username from system or prompt user"""
    # Try to get system username
    try:
        username = os.getlogin()
    except:
        username = platform.node()  # Fallback to computer name
    
    # Allow user to override
    custom_username = st.text_input("Username", value=username, disabled=True)
    return custom_username

def user_chat_page_v1():
    st.title("📄  Document Chat Assistant")
    
    # Get username
    username = get_username()
    
    # knowledge base Selection   
    vector_db_dir = 'vector_databases'
    vector_dbs = [f for f in os.listdir(vector_db_dir) if f.endswith('_index.faiss')]
    
    if not vector_dbs:
        st.warning("No knowledge bases available. Please create one in the Admin Page.")
        return
    
    selected_db = st.selectbox("Select knowledge base", vector_dbs)
    
    # Initialize chat and tracking
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'interaction_ids' not in st.session_state:
        st.session_state.interaction_ids = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                db_path = os.path.join(vector_db_dir, selected_db)
                assistant = EnhancedDocumentAssistant(db_path)
                
                # Generate response with tracking
                response_data = assistant.generate_response(prompt, username)
                response = response_data['response']
                interaction_id = response_data['interaction_id']
                
                # Store interaction ID
                st.session_state.interaction_ids.append(interaction_id)
                
                # Display response with feedback buttons
                st.markdown(response)
                
                # Feedback mechanism
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful", key=f"like_{interaction_id}"):
                        assistant.tracker.log_feedback(interaction_id, 'like','ok')
                        st.success("Thank you for your feedback!")
                
                with col2:
                    if st.button("👎 Not Helpful", key=f"dislike_{interaction_id}"):
                        reason = st.text_input("Why wasn't this helpful?", key=f"reason_{interaction_id}")
                        assistant.tracker.log_feedback(interaction_id, 'dislike', reason)
                        st.success("Thank you for your feedback!")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Analytics Dashboard
    if st.sidebar.button("View Analytics"):
        tracker = UserActivityTracker()
        analytics = tracker.get_analytics()
        
        st.sidebar.header("🔍 Interaction Analytics")
        st.sidebar.metric("Total Interactions", analytics['total_interactions'])
        st.sidebar.metric("Avg Response Time", f"{analytics['avg_response_time']:.2f} seconds")
        
        st.sidebar.subheader("Feedback Distribution")
        for feedback_type, count, percentage in analytics['feedback_stats']:
            st.sidebar.metric(
                f"{feedback_type.capitalize()} Feedback", 
                f"{count} ({percentage}%)"
            )

if __name__ == "__main__":
    user_chat_page_v1()
