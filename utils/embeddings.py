from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeEmbeddings

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings


class EmbeddingsFactory:
    @staticmethod
    def create_openai_embeddings(model_name: str, api_key: str) -> Embeddings:
        return OpenAIEmbeddings(
            model_name=model_name,
            openai_api_key=api_key,
        )

    @staticmethod
    def create_huggingface_embeddings(model_name: str) -> Embeddings:
        return HuggingFaceEmbeddings(model_name=model_name)

    @staticmethod
    def create_llama_cpp_embeddings(model_path: str) -> Embeddings:
        return LlamaCppEmbeddings(model_path=model_path)

    @staticmethod
    def create_pinecone_embeddings() -> Embeddings:
        return PineconeEmbeddings(model="multilingual-e5-large")
