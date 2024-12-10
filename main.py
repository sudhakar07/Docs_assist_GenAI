import os
import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
from dotenv import load_dotenv

# Import custom modules
from admin_page import admin_page

from user_chat_page_v1 import user_chat_page_v1
from doc_assist_with_highlight import response_highlight

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key_secrectpass = st.secrets["api_key"]
genai.configure(api_key=api_key_secrectpass)

def main():
   
    # Add navigation
    page = st.sidebar.radio("Navigate", ["Admin Page", "Document Assist", "DocAssist with Field-Highlight"])
    st.sidebar.info("Developed by - Sudhakar G.")
    st.sidebar.info("Only for Learning Purpose.")
    
    if page == "Admin Page":
        admin_page()
    if page == "Document Assist":
        user_chat_page_v1()
    if page == "DocAssist with Field-Highlight":
        response_highlight()
    # else:
    #     user_chat_page()

if __name__ == "__main__":
    genai.configure(api_key=api_key_secrectpass)
    main()
    
    hide_st_style = """ <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            ._profileContainer_gzau3_53 {visibility: hidden;}
            _profilePreview_gzau3_63 {visibility: hidden;}
            </style>"""

    st.markdown(hide_st_style, unsafe_allow_html=True)
