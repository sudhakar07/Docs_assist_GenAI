import os
import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
from dotenv import load_dotenv

# Import custom modules
from admin_page import admin_page

from user_chat_page_v1 import user_chat_page_v1

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key_secrectpass = st.secrets["api_key"]
genai.configure(api_key=api_key_secrectpass)

def main():
    st.set_page_config(page_title="Document Assist", layout="wide")
    
    # Add navigation
    page = st.sidebar.radio("Navigate", ["Admin Page", "Document Assist"])
    st.sidebar.write("Developed by - Sudhakar G.")
    st.sidebar.info("Only for Learning Purpose.")
    
    if page == "Admin Page":
        admin_page()
    if page == "Document Assist":
        user_chat_page_v1()
    # else:
    #     user_chat_page()

if __name__ == "__main__":
    genai.configure(api_key=api_key_secrectpass)
    main()
