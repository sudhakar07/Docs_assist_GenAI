# Document Assist: Multi-PDF RAG Chatbot

## Overview
Document Assist is a Streamlit-based generative AI application that allows administrators to upload and manage PDF documents, and users to interactively chat with these documents using Retrieval Augmented Generation (RAG).

## Features
- Admin Page:
  - Secure login
  - PDF upload and vector database creation
  - Vector database management
- User Chat Page:
  - Select from available vector databases
  - Chat with documents
  - Retrieve contextually relevant information

## Prerequisites
- Python 3.8+
- Gemini API Key

## Installation
1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your Gemini API key
5. Run the application:
   ```
   streamlit run main.py
   ```

## Usage
- Admin Credentials:
  - Username: admin
  - Password: 
- Upload PDFs in the Admin Page
- Chat with documents in the User Chat Page

## Technologies
- Streamlit
- Google Generative AI (Gemini)
- FAISS
- LangChain