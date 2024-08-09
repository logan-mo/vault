# PineCone Retriever
# Chroma Retriever
# LanceDB Retriever
# Semantic Search + Hybrid Search


class RetrieverFactory:
    @staticmethod
    def create_pinecone_retriever(): ...

    @staticmethod
    def create_chroma_db_retriever(): ...

    @staticmethod
    def create_lance_db_retriever(): ...
