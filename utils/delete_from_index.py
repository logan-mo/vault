from pinecone import Pinecone

def delete_all_from_index(pc: Pinecone, index_name: str):
    try:
        index = pc.Index(index_name)
        index.delete(delete_all=True)
        print(f"All vectors deleted from index '{index_name}'")
    except Exception as e:
        print(f"Error deleting vectors from index '{index_name}': {str(e)}")
        