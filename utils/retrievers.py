# PineCone Retriever
# Chroma Retriever
# LanceDB Retriever
# Semantic Search + Hybrid Search

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever


from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb import LanceDBConnection
from lancedb.rerankers import LinearCombinationReranker
from langchain_community.vectorstores import LanceDB

import openai


def LanceDBHybridSearchRetriever(BaseRetriever):

    def __init__(self, db: LanceDBConnection, index_name: str, openai_api_key: str):
        super().__init__()
        self.db = db

        openai.api_key = openai_api_key
        embeddings = get_registry().get("openai").create()

        class LanceDocuments(LanceModel):
            vector: Vector(embeddings.ndims()) = embeddings.VectorField()
            text: str = embeddings.SourceField()

        if index_name not in db.table_names():
            db.create_table(index_name, LanceDocuments)
        self.table = db.open_table(index_name)

        self.reranker = LinearCombinationReranker(weight=0.3)

    def add_documents(self, documents: list[dict]) -> None:
        self.table.add({"text": doc} for doc in documents)
        self.table.create_fts_index("text")

    def _get_relevant_documents(
        self,
        query: str,
    ):
        results = (
            self.table.search(query, query_type="hybrid")
            .rerank(reranker=self.reranker)
            .to_pandas()
        )
        for idx, result in results:
            results[idx] = Document(page_content=result["text"])

        return results


# Self Query retriever https://python.langchain.com/v0.2/docs/integrations/retrievers/self_query/pinecone/
class RetrieverFactory:
    @staticmethod
    def create_pinecone_retriever(
        pinecone_api_key: str,
        embeddings: Embeddings,
        index_name: str,
        namespace: str,
        cloud: str | None = "aws",
        region: str | None = "us-west-1",
        metric: str | None = "cosine",
    ):
        pc = Pinecone(api_key=pinecone_api_key)

        embedding_dimension = len(embeddings.embed_query("Hi"))

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=embedding_dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        # vectorstore = PineconeVectorStore(
        #     index_name=index_name,
        #     embedding=embeddings,
        #     namespace=namespace,
        # )

        bm25_encoder = BM25Encoder().default()
        retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings, sparse_encoder=bm25_encoder, index=pc
        )

        return retriever

    @staticmethod
    def create_lance_db_retriever(openai_api_key: str, index_name: str = "./.lancedb"):
        db_conn = LanceDBConnection(index_name)
        return LanceDBHybridSearchRetriever(
            db=db_conn, index_name=index_name, openai_api_key=openai_api_key
        )
