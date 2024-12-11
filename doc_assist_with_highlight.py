import pandas as pd
import streamlit as st
import os
import fitz
import tempfile
# from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQA

import io
from streamlit_pdf_viewer import pdf_viewer
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import faiss
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
api_key_secrectpass = st.secrets["api_key"]
genai.configure(api_key=api_key_secrectpass)
Groq_api_key_secrectpass = st.secrets["GROQ_API_KEY"]





st.subheader("Upload a document to get started.")


# Custom function to extract document objects from uploaded file
def extract_documents_from_file(uploaded_file):
    # with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    #     temp_file.write(uploaded_file.read())
    
    # loader = PyPDFLoader(temp_file.name)
    # documents = loader.load()
    
    # os.unlink(temp_file.name)  # Clean up the temporary file
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
       file.write(uploaded_file.getvalue())
       file_name = uploaded_file.name

    loader = PyPDFLoader(temp_file)
    documents = loader.load_and_split()


    return documents


def locate_pages_containing_excerpts(document, excerpts):
    relevant_pages = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        if any(page.search_for(excerpt) for excerpt in excerpts):
            relevant_pages.append(page_num)
    return relevant_pages if relevant_pages else [0]



def initialize_language_model():
    return ChatGroq(
        temperature=0,
        groq_api_key=Groq_api_key_secrectpass,
        model_name="mixtral-8x7b-32768"
    )


# @st.cache_resource
# def get_embeddings():
#     embeddings = OllamaEmbeddings(
#         model="nomic-embed-text",
#         base_url="http://localhost:11434"  # Adjust this URL if Ollama is running on a different address
#     )
#     return embeddings



def get_embeddings_google():
    genai.configure(api_key=api_key_secrectpass)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return embeddings


def generate_embeddings(_texts_ext):
        genai.configure(api_key='')
        # """
        # Generate embeddings using Gemini Pro API
        
        # Args:
        #     texts (List[str]): List of text chunks to embed
        
        # Returns:
        #     List[List[float]]: List of embedding vectors
        # """
        model="models/text-embedding-004"
        embeddings = []
        # st.write(_texts_ext)
        for text_ext in _texts_ext:
            try:
                # Use embed_content method
                result = genai.embed_content(
                    model=model,
                    content=text_ext.page_content,
                    task_type="retrieval_document"
                )
                # st.write(text_ext)
                # st.write(result['embedding'])
                embeddings.append(result['embedding'])
            except Exception as e:
                st.error(f"Embedding error: {e}")
                # Append a zero vector if embedding fails
                # embeddings.append([0.0] * 768)  # Assuming 768-dimensional embedding
        
        st.write(embeddings)
        return embeddings



@st.cache_resource
def setup_qa_system(_documents_ext):
    try:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # text_splitter = RecursiveCharacterTextSplitter(
        #             chunk_size=1000,
        #             chunk_overlap=200,
        #             length_function=len
        #         )
        text_chunks = text_splitter.split_documents(_documents_ext)
        st.write(_documents_ext)
        # embeddings = generate_embeddings(text_chunks)
        vector_store = FAISS.from_documents(text_chunks, get_embeddings_google())
        # vector_store = FAISS.from_documents(text_chunks,  generate_embeddings(text_chunks))
        # st.info("vector_store")
        retriever = vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
        )
        # st.info("retriever")
        return RetrievalQA.from_chain_type(
            initialize_language_model(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": CUSTOM_PROMPT},
        )
    except Exception as e:
        st.error(f"Error setting up QA system: {str(e)}")
        
        return None


def generate_highlight_annotations(document, excerpts):
    annotations = []
    for page_num, page in enumerate(document):
        for excerpt in excerpts:
            for inst in page.search_for(excerpt):
                annotations.append({
                    "page": page_num + 1,
                    "x": inst.x0,
                    "y": inst.y0,
                    "width": inst.x1 - inst.x0,
                    "height": inst.y1 - inst.y0,
                    "color": "red",
                })
    return annotations


CUSTOM_PROMPT_TEMPLATE = """
Use the following pieces of context to answer the user question. If you
don't know the answer, just say that you don't know, don't try to make up an
answer.

{context}

Question: {question}

Please provide your answer in the following JSON format: 
{{
    "answer": "Your detailed answer here",
    "sources": "Direct sentences or paragraphs from the context that support 
        your answers. ONLY RELEVANT TEXT DIRECTLY FROM THE DOCUMENTS. DO NOT 
        ADD ANYTHING EXTRA. DO NOT INVENT ANYTHING."
}}

The JSON must be a valid json format and can be read with json.loads() in
Python. Answer:
"""

CUSTOM_PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"]
)

def response_highlight():

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        file = uploaded_file.read()

        with st.spinner("Processing file..."):
            documents = extract_documents_from_file(uploaded_file)
            # st.write(documents)
            st.session_state.doc = fitz.open(stream=io.BytesIO(file), filetype="pdf")

        if documents:
            with st.spinner("Setting up QA system..."):
                qa_system = setup_qa_system(documents)
                if qa_system is None:
                    st.error("Failed to set up QA system. Please check if Ollama is running and try again.")
                else:
                    st.success("Doc Assist  App ready!")
                    # Continue with the rest of your chat logic here

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [
                    {"role": "assistant", "content": "Hello! How can I assist you today? "}
                ]

            for msg in st.session_state.chat_history:
                st.chat_message(msg["role"]).write(msg["content"])

            if user_input := st.chat_input("Your message"):
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input}
                )
                st.chat_message("user").write(user_input)

                with st.spinner("Generating response..."):
                    try:
                        result = qa_system.invoke({"query": user_input})
                        sources_res= False
                        answer_res= True
                        # st.write(json.loads(result['result']))

                        try:
                            parsed_result = json.loads(result['result'])

                            answer = parsed_result.get('answer')
                        
                            sources = parsed_result.get('sources')
                            sources_res=True
                        except:
                            st.error("No Source.")
                            sources_res= False
                            parsed_result = json.loads(result['result'])
                            answer = parsed_result.get('answer')

                        # answer = "test"
                        # sources = "Our findings indicate that the importance of science and critical thinking skills are strongly negatively associated with exposure, suggesting that occupations requiring these skills are less likely to be impacted by current LLMs. Conversely, programming and writing skills show a strong positive association with exposure, implying that occupations involving these skills are more susceptible to being influenced by LLMs."
                        if answer_res is True:
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": answer}
                            )
                            st.chat_message("assistant").write(answer)

                        if ((sources_res is  True) and (str(answer).find("does not") == -1)) :
                            sources = sources.split(". ") if pd.notna(sources) else []
                            st.session_state.sources = sources
                            st.session_state.chat_occurred = True

                            if file and st.session_state.get("chat_occurred", False):
                                doc = st.session_state.doc
                                st.session_state.total_pages = len(doc)
                                if "current_page" not in st.session_state:
                                    st.session_state.current_page = 0

                                pages_with_excerpts = locate_pages_containing_excerpts(doc, sources)

                                if "current_page" not in st.session_state:
                                    st.session_state.current_page = pages_with_excerpts[0]

                                st.session_state.cleaned_sources = sources
                                st.session_state.pages_with_excerpts = pages_with_excerpts

                                st.markdown("### PDF Preview with Highlighted Excerpts")

                                col1, col2, col3 = st.columns([1, 3, 1])
                                with col1:
                                    if st.button("Previous Page") and st.session_state.current_page > 0:
                                        st.session_state.current_page -= 1
                                with col2:
                                    st.write(
                                        f"Page {st.session_state.current_page + 1} of {st.session_state.total_pages}"
                                    )
                                with col3:
                                    if (
                                        st.button("Next Page")
                                        and st.session_state.current_page
                                        < st.session_state.total_pages - 1
                                    ):
                                        st.session_state.current_page += 1

                                annotations = generate_highlight_annotations(doc, st.session_state.sources)

                                if annotations:
                                    first_page_with_excerpts = min(ann["page"] for ann in annotations)
                                else:
                                    first_page_with_excerpts = st.session_state.current_page + 1

                                pdf_viewer(
                                    file,
                                    width=700,
                                    height=800,
                                    annotations=annotations,
                                    pages_to_render=[first_page_with_excerpts],
                                )

                    except json.JSONDecodeError:
                        st.error(
                            "There was an error parsing the response. Please try again."
                        )


if __name__ == "__main__":
    
    response_highlight()
