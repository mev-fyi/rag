import logging
import os
from datetime import datetime

from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores import PineconeVectorStore

from rag.Llama_index_sandbox import index_dir
from rag.Llama_index_sandbox.utils import timeit
import pinecone

api_key = os.environ["PINECONE_API_KEY"]

@timeit
def initialise_vector_store(embedding_model_chunk_size) -> PineconeVectorStore:
    pinecone.init(api_key=api_key, environment=os.environ["PINECONE_API_ENVIRONMENT"])
    index_name = "quickstart"

    # Check if the index already exists
    existing_indexes = pinecone.list_indexes()
    if index_name in existing_indexes:
        # If the index exists, delete it
        pinecone.delete_index(index_name)

    # Create a new index
    pinecone.create_index(name=index_name, dimension=embedding_model_chunk_size, metric="cosine", pod_type="p1")
    pinecone_index = pinecone.Index(index_name=index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Optionally, you might want to delete all contents in the index
    # pinecone_index.delete(deleteAll=True)
    return vector_store


@timeit
def persist_index(index, embedding_model_name, chunk_size, chunk_overlap):
    """
    Persist the index to disk.
    NOTE: Given that we use an external DB, this only writes a json containing the ID referring to that DB.
    """
    try:
        # Format the filename
        date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
        name = f"{date_str}_{embedding_model_name}_{chunk_size}_{chunk_overlap}"
        persist_dir = index_dir + name
        # check if index_dir and if not create it
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        # NOTE 2023-09-29: https://stackoverflow.com/questions/76837143/llamaindex-index-storage-context-persist-not-storing-vector-store
        #   Vector Store IS NOT persisted. The method index.storage_context.persist is failing silently since when attempting to
        #   load the index back, it fails since there is no vector json file
        index.storage_context.persist(persist_dir=persist_dir)  # TODO 2023-09-29: understand why vector_store is not saved there
        # create a vector_store.json file with {} inside
        with open(f"{persist_dir}/vector_store.json", "w") as f:
            f.write("{}")

        logging.info(f"Successfully persisted index {persist_dir} to disk.")
    except Exception as e:
        logging.error(f"Failed to persist index to disk. Error: {e}")


@timeit
def load_nodes_into_vector_store_create_index(nodes, embedding_model_chunk_size) -> VectorStoreIndex:
    """
    We now insert these nodes into our PineconeVectorStore.

    NOTE: We skip the VectorStoreIndex abstraction, which is a higher-level
    abstraction that handles ingestion as well. We use VectorStoreIndex in the next section to fast-track retrieval/querying.
    """
    vector_store = initialise_vector_store(embedding_model_chunk_size=embedding_model_chunk_size)
    vector_store.add(nodes)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index


@timeit
def load_index_from_disk() -> VectorStoreIndex:
    # load the latest directory in index_dir
    persist_dir = f"{index_dir}{sorted(os.listdir(index_dir))[-1]}"
    logging.info(f"LOADING INDEX {persist_dir} FROM DISK")
    try:
        pinecone.init(api_key=api_key, environment=os.environ["PINECONE_API_ENVIRONMENT"])
        index_name = "quickstart"
        vector_store = PineconeVectorStore(pinecone_index=pinecone.Index(index_name))
        index = VectorStoreIndex.from_vector_store(vector_store)
        return index
    except Exception as e:
        logging.error(f"Error: {e}")
        # To accommodate for the case where the vector_store.json file is not persisted https://stackoverflow.com/questions/76837143/llamaindex-index-storage-context-persist-not-storing-vector-store
        if "No existing llama_index.vector_stores.simple" in str(e):
            # create a vector_store.json file with {} inside
            with open(f"{persist_dir}/vector_store.json", "w") as f:
                f.write("{}")
            try:
                pinecone.init(api_key=api_key, environment=os.environ["PINECONE_API_ENVIRONMENT"])
                index_name = "quickstart"
                vector_store = PineconeVectorStore(pinecone_index=pinecone.Index(index_name))
                index = VectorStoreIndex.from_vector_store(vector_store)
                return index
            except Exception as e:
                logging.error(f"Error: {e}")
                exit(1)



