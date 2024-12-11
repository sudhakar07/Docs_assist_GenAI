import streamlit as st  # Streamlit is used to create the web app interface
from PyPDF2 import PdfReader  # PyPDF2 is used to read PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into manageable chunks
import io
# pip install -U langchain-community
from langchain_community.vectorstores import FAISS  # Vector store to store and retrieve text embeddings using FAISS
from langchain_community.document_loaders import TextLoader, CSVLoader  # Loaders for text and CSV documents
from langchain_community.embeddings import OllamaEmbeddings  # Creates embeddings for text using different models
from langchain_community.llms import Ollama  # Large Language Model interface for Ollama

from langchain.prompts import ChatPromptTemplate  # Creates templates for prompts to the LLM
from langchain.chains import RetrievalQA  # Creates a QA chain using retrieval-based methods
from langchain.schema import Document  # Document schema for text data
import os  # OS module for environment configuration and file handling
import tempfile  # Create temporary files
import json  # JSON module for reading and writing JSON data
from datetime import datetime  # Handles date and time operations
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq


api_key_secrectpass = st.secrets["api_key"]
genai.configure(api_key=api_key_secrectpass)
Groq_api_key_secrectpass = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = Groq_api_key_secrectpass

# Function to configure proxy settings if needed
def configure_proxy(use_proxy):
    # proxy = "http://proxy.my-company.com:8080" if use_proxy else ""
    # os.environ['http_proxy'] = proxy
    # os.environ['https_proxy'] = proxy
    pass

# Read data from uploaded files
def read_data(files, loader_type):
    documents = []
    for file in files:
        
        with file as f:
            
            # st.write(tmp_file_path)
            try:
                if loader_type == "PDF":
                    pdf_reader = PdfReader(f)
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        documents.append(Document(page_content=text, metadata={"source": f.name, "page_number": page_num + 1}))
                elif loader_type == "Text":
                    loader = TextLoader(f)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = f.name
                    documents.extend(docs)
                elif loader_type == "CSV":
                    loader = CSVLoader(f)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = f.name
                    documents.extend(docs)
            finally:
                #os.remove(tmp_file_path)
                st.write("")

    st.write(documents)
    return documents

# Split text into chunks
def get_chunks(texts, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for text in texts:
        split_texts = text_splitter.split_text(text.page_content)
        for split_text in split_texts:
            chunks.append(Document(page_content=split_text, metadata=text.metadata))
            
    st.write(chunks)
    return chunks

def get_embeddings_google():
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return embeddings

# Store text chunks in a vector store using FAISS
def vector_store(text_chunks, embedding_model_name, vector_store_path):
    # embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=embedding_model_name)
    embeddings = get_embeddings_google()
    vector_store = FAISS.from_texts(texts=[doc.page_content for doc in text_chunks], embedding=embeddings, metadatas=[doc.metadata for doc in text_chunks])
    vector_store.save_local(vector_store_path)

# Load the vector store using FAISS
def load_vector_store(embedding_model_name, vector_store_path):
    # embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=embedding_model_name)
    embeddings = get_embeddings_google()
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Save conversation history to a JSON file
def save_conversation(conversation, vector_store_path):
    conversation_path = os.path.join(vector_store_path, "conversation_history.json")
    with open(conversation_path, "w") as f:
        json.dump(conversation, f, indent=4)

# Load conversation history from a JSON file
def load_conversation(vector_store_path):
    conversation_path = os.path.join(vector_store_path, "conversation_history.json")
    if os.path.exists(conversation_path):
        with open(conversation_path, "r") as f:
            conversation = json.load(f)
    else:
        conversation = []
    return conversation

# Convert Document object to a serializable format
def document_to_dict(doc):
    return {
        "metadata": doc.metadata
    }

@st.cache_resource
def initialize_language_model():
    
    return ChatGroq(
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="mixtral-8x7b-32768"
    )
    # model = genai.GenerativeModel('gemini-pro')
    return model

# Get a conversational chain response from the LLM
def get_conversational_chain(retriever, ques, llm_model, system_prompt):
    # llm = Ollama(model=llm_model, verbose=True)
    llm = initialize_language_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True  # Return source documents
    )
    response = qa_chain.invoke({"query": ques})
    return response

# Handle user input and display the response
def user_input(user_question, embedding_model_name, vector_store_path, num_docs, llm_model, system_prompt):
    vector_store = load_vector_store(embedding_model_name, vector_store_path)
    retriever = vector_store.as_retriever(search_kwargs={"k": num_docs})
    response = get_conversational_chain(retriever, user_question, llm_model, system_prompt)
    
    conversation = load_conversation(vector_store_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if 'result' in response:
        result = response['result']
        source_documents = response['source_documents'] if 'source_documents' in response else []
        conversation.append({
            "question": user_question, 
            "answer": result, 
            "timestamp": timestamp, 
            "llm_model": llm_model,
            "source_documents": [document_to_dict(doc) for doc in source_documents]
        })
        st.write("Reply: ", result)
        st.write(f"**LLM Model:** {llm_model}")
        
        st.write("### Source Documents")
        for doc in source_documents:
            metadata = doc.metadata
            st.write(f"**Source:** {metadata.get('source', 'Unknown')}, **Page Number:** {metadata.get('page_number', 'N/A')}, **Additional Info:** {metadata}")
        st.markdown("<hr style='border:1px solid gray;'>", unsafe_allow_html=True)
    else:
        conversation.append({"question": user_question, "answer": response, "timestamp": timestamp, "llm_model": llm_model})
        st.write("Reply: ", response)
    
    save_conversation(conversation, vector_store_path)
    
    st.write("### Conversation History")
    for entry in sorted(conversation, key=lambda x: x['timestamp'], reverse=True):
        st.write(f"**Q ({entry['timestamp']}):** {entry['question']}")
        st.write(f"**A:** {entry['answer']}")
        st.write(f"**LLM Model:** {entry['llm_model']}")
        if 'source_documents' in entry:
            for doc in entry['source_documents']:
                st.write(f"**Source:** {doc['metadata'].get('source', 'Unknown')}, **Page Number:** {doc['metadata'].get('page_number', 'N/A')}, **Additional Info:** {doc['metadata']}")  # Display source filename, page number, and additional metadata
        st.markdown("<hr style='border:1px solid gray;'>", unsafe_allow_html=True)

# Main function to run the Streamlit app
def main_web_intract():
    
    st.header("Chat with Your Files using Llama3 or Mistral")

    # Add GitHub link below the header
   
    st.sidebar.markdown(
        """
        <div style="text-align: center;">
            <img src="https://ollama.com/public/ollama.png" alt="Ollama Logo" style="width: 50px; height: auto;">
            <p><b>Ollama Playground AI with GROQ</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add system prompt input field
    system_prompt = st.sidebar.text_area("System Prompt", value="You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in the provided context, just say, 'answer is not available in the context', don't provide the wrong answer")

    user_question = st.text_input("Ask a Question from the Files")
    use_proxy = st.sidebar.checkbox("Use Proxy", value=False)
    configure_proxy(use_proxy)

    embedding_model_name = st.sidebar.selectbox(
        "Select Embedding Model",
        ["mxbai-embed-large", "llama3:instruct", "mistral:instruct"]
    )

    llm_model = st.sidebar.selectbox(
        "Select LLM Model",
        ["llama3:instruct", "mistral:instruct"]
    )

    vector_store_path = st.sidebar.text_input("Vector Store Path (will reload if there)", "my_store")
    data_type = st.sidebar.selectbox(
        "Select Data Type",
        ["PDF", "Text", "CSV"]
    )

    chunk_text = st.sidebar.checkbox("Chunk Text", value=True)
    chunk_size = st.sidebar.number_input("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.sidebar.number_input("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    num_docs = st.sidebar.number_input("Number of Documents to Retrieve", min_value=1, max_value=10, value=3, step=1)

    if user_question:
        user_input(user_question, embedding_model_name, vector_store_path, num_docs, llm_model, system_prompt)

    with st.sidebar:
        st.title("Documents:")
        data_files = st.file_uploader("Upload your Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_documents = read_data(data_files, data_type)
                if chunk_text:
                    text_chunks = get_chunks(raw_documents, chunk_size, chunk_overlap)
                else:
                    text_chunks = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in raw_documents]
                vector_store(text_chunks, embedding_model_name, vector_store_path)
                st.success("Done")
    
    # Footer with three columns
    st.markdown("<hr>", unsafe_allow_html=True)
    

if __name__ == "__main__":
    main_web_intract()
