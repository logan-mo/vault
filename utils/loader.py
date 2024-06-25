import os
import streamlit as st
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec

def initialize_pinecone():
    try:
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        st.success("Successfully connected to Pinecone")
        return pc
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        st.stop()

def index_exists(pc, index_name):
    if index_name not in pc.list_indexes().names():
        st.warning(f"Index '{index_name}' does not exist. Creating it now...")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud=st.secrets["PINECONE_CLOUD"],
                region=st.secrets["PINECONE_REGION"]
            )
        )
        st.success(f"Created index '{index_name}'")

def load_and_index_docs(docs, embeddings, index_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    # Extract text content and metadata
    texts_with_metadata = [(t.page_content, t.metadata) for t in texts]

    # Index texts with metadata
    LangchainPinecone.from_texts(
        [t[0] for t in texts_with_metadata],  # Text content
        embeddings,
        index_name=index_name,
        metadatas=[t[1] for t in texts_with_metadata]  # Metadata
    )

def load_vault():
    directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
    loader = DirectoryLoader(directory, glob="**/*", loader_cls=UnstructuredFileLoader, use_multithreading=True, silent_errors=True, show_progress=True)
    return loader.load()

# def load_vault():
#     directory = r"C:\vault\app\scan"
#     loader = DirectoryLoader(directory, glob="**/*", loader_cls=UnstructuredFileLoader, use_multithreading=True, silent_errors=True, show_progress=True)
#     return loader.load()

def load_file(uploaded_file):
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
    loader = UnstructuredFileLoader(temp_file_path)
    os.remove(temp_file_path)
    return loader.load()

def load_directory(directory):
    loader = DirectoryLoader(directory, glob="**/*", loader_cls=UnstructuredFileLoader, use_multithreading=True, silent_errors=True, show_progress=True)
    return loader.load()

def retrieve_documents(query, embeddings, index_name, k=3):
    vectorstore = LangchainPinecone(index_name=index_name, embedding=embeddings)
    docs = vectorstore.similarity_search(query, k=k)
    return docs