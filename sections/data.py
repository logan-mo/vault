import streamlit as st
from langchain_openai import OpenAIEmbeddings
from utils.loader import (
    initialize_pinecone,
    index_exists,
    load_and_index_docs,
    load_vault,
    load_file,
    load_directory
)
from utils.delete_from_index import delete_all_from_index


def show():
    st.title("Load Data")

    pc = initialize_pinecone()
    index_name = st.secrets["PINECONE_INDEX"]
    index_exists(pc, index_name)

    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

    if st.button("Load VAULT"):
        with st.spinner("Loading and indexing the VAULT..."):
            delete_all_from_index(pc, index_name)
            st.success(f"Cleaned index, loading now...")
            docs = load_vault()
            load_and_index_docs(docs, embeddings, index_name)
        st.success(f"Loaded and indexed {len(docs)} documents from the VAULT.")

    uploaded_file = st.file_uploader("Choose a file to upload")
    if st.button("Load File") and uploaded_file is not None:
        with st.spinner("Loading and indexing file..."):
            docs = load_file(uploaded_file)
            load_and_index_docs(docs, embeddings, index_name)
        st.success(f"Loaded and indexed {uploaded_file.name}")

    directory = st.text_input("Enter directory path to load:")
    if st.button("Load Directory") and directory:
        with st.spinner("Loading and indexing directory..."):
            docs = load_directory(directory)
            load_and_index_docs(docs, embeddings, index_name)
        st.success(f"Loaded and indexed {len(docs)} documents from {directory}")

if __name__ == "__main__":
    show()