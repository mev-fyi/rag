import logging
import os
from datetime import datetime

from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores import PineconeVectorStore

from rag.Llama_index_sandbox import index_dir
from rag.Llama_index_sandbox.utils import timeit


@timeit
def initialise_vector_store(dimension):
    import pinecone
    api_key = os.environ["PINECONE_API_KEY"]
    pinecone.init(api_key=api_key, environment=os.environ["PINECONE_API_ENVIRONMENT"])

    index_name = "quickstart"

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
def persist_index(index, embedding_model_name, chunk_size, chunk_overlap):
    """
    Persist the index to disk.
    """
    try:
        # Format the filename
        date_str = datetime.now().strftime("%Y-%m-%d")
        name = f"{date_str}_{embedding_model_name}_{chunk_size}_{chunk_overlap}"
        persist_dir = index_dir + name
        # check if index_dir and if not create it
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        index.storage_context.persist(persist_dir=persist_dir)
        logging.info("Successfully persisted index to disk.")
    except Exception as e:
        logging.error(f"Failed to persist index to disk. Error: {e}")


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


def load_index_from_disk():
    try:
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        return load_index_from_storage(storage_context)
    except FileNotFoundError:
        logging.error("No index found. Please run the index creation script first.")
        exit(1)



