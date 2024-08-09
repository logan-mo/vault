from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.llms import BaseLLM

from langchain.chains import create_retrieval_chain
from langchain.prompts import Prompt, PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_cohere import CohereRerank

from utils.retrievers import RetrieverFactory


def create_stuff_document_chain(
    llm: BaseLLM, retriever: BaseRetriever, reranker_type: str = "Cohere"
) -> callable:

    reranked = (
        CohereRerank(model="rerank-english-v3.0")
        if reranker_type == "Cohere"
        else LLMChainFilter.from_llm(llm=llm)
    )
    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=reranked, base_retriever=retriever
    )

    history_aware_retriever = RetrieverFactory.create_history_aware_prompt(
        llm, reranked_retriever
    )

    # TODO: Translate the system prompt to German
    SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Also if the context has no information regarding the user's question, apologize to the user about being unable to find any laws or statures related to the users' query. If you know the answer, take a deep breath and explain your reasoning."
    query_template = "<context>\n {context}\n</context>\n" "\nQuestion: {input}"

    template = f"{SYSTEM_PROMPT}\n\n{query_template}"

    prompt = PromptTemplate(template=template, input_variables=["context", "input"])

    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain,
    )
    return retrieval_chain
