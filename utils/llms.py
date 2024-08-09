from langchain_core.language_models.llms import BaseLLM

from langchain_openai import ChatOpenAI
from langchain_community.llms.vllm import VLLM
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


class LLMFactory:
    @staticmethod
    def create_openai_llm(model_name: str, api_key: str) -> BaseLLM:
        return ChatOpenAI(
            model_name=model_name,
            openai_api_key=api_key,
        )

    # TODO: Test on cloud
    @staticmethod
    def create_huggingface_llm(model_name: str) -> BaseLLM:
        return HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
        )

    # TODO: Test on cloud
    @staticmethod
    def create_vllm_llm(model_name: str) -> BaseLLM:
        return VLLM(
            model_name=model_name,
            trust_remote_code=True,
            max_tokens=2000,
            top_k=1,
            temperature=0.3,
        )

    @staticmethod
    def create_llama_cpp_llm(model_path: str) -> BaseLLM:
        """
        Parameters
        ----------
        model_path : str
            The path to the LlamaCpp model file (gguf or ggml binary file)
        e.g:

        """
        return LlamaCpp(
            model_path=model_path,
            temperature=0.3,
            max_tokens=2000,
            top_p=1,
            verbose=False,
        )
