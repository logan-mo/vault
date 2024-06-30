import cohere
import streamlit as st

# import toml

# config = toml.load('app/.streamlit/secrets.toml')
# co = cohere.Client(config["COHERE_API_KEY"])

co = cohere.Client(st.secrets["COHERE_API_KEY"])

def rerank_chunks(query, chunks, top_n=3):
    try:
        # Run the rerank model 
        results = co.rerank(query=query, documents=chunks, top_n=top_n, model="rerank-multilingual-v3.0")

        # Extract the top_n results based on relevance_score
        top_chunks = sorted(results.results, key=lambda x: x.relevance_score, reverse=True)[:top_n]
        
        # Return the top chunks as a list, using the index to get the document from chunks
        return [chunks[chunk.index] for chunk in top_chunks if chunk.index < len(chunks)]
    except cohere.errors.BadRequestError as e:
        st.error("An error occurred while reranking the chunks. Please check your input and try again.")
        return []