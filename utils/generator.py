import openai
import streamlit as st
from pinecone import Pinecone 
from utils.reranker import rerank_chunks


def get_context(query):
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index(st.secrets["PINECONE_INDEX"])

    embed_model = "text-embedding-ada-002"
    result = openai.embeddings.create(
        input=[query],
        model=embed_model
    )

    embedding = result.data[0].embedding

    result = index.query(vector=embedding, top_k=10, include_metadata=True)

    chunks = [item['metadata']['text'] for item in result['matches']]

    reranked_chunks = rerank_chunks(query, chunks)

    context = "\n\n---\n\n".join(reranked_chunks)+"\n\n-----\n\n"

    augmented_query = "<context> \n\n" + context + "</context> \n\n" + query 
    
    return augmented_query