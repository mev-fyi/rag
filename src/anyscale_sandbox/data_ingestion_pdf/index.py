import os

from llama_index import VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore

from src.anyscale_sandbox.utils import timeit


@timeit
def initialise_vector_store(dimension):
    import pinecone
    api_key = os.environ["PINECONE_API_KEY"]
    pinecone.init(api_key=api_key, environment=os.environ["PINECONE_API_ENVIRONMENT"])

    index_name = "mevfyi"

    # Check if the index already exists
    existing_indexes = pinecone.list_indexes()
    if index_name in existing_indexes:
        # If the index exists, delete it
        pinecone.delete_index(index_name)

    # Create a new index
    pinecone.create_index(name=index_name, dimension=dimension, metric="cosine", pod_type="p1")
    pinecone_index = pinecone.Index(index_name=index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Optionally, you might want to delete all contents in the index
    # pinecone_index.delete(deleteAll=True)
    return vector_store


@timeit
def load_nodes_into_vector_store_create_index(nodes, vector_store):
    """
    We now insert these nodes into our PineconeVectorStore.

    NOTE: We skip the VectorStoreIndex abstraction, which is a higher-level
    abstraction that handles ingestion as well. We use VectorStoreIndex in the next section to fast-track retrieval/querying.
    """
    vector_store.add(nodes)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index
