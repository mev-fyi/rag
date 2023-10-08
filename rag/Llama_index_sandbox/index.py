from llama_index import VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore
import logging
import os
from datetime import datetime
import pinecone
from pathlib import Path

from rag.Llama_index_sandbox.data_ingestion_youtube.load.load import load_video_transcripts
from rag.Llama_index_sandbox import pdfs_dir, video_transcripts_dir
import rag.Llama_index_sandbox.data_ingestion_pdf.load as load_pdf
import rag.Llama_index_sandbox.data_ingestion_pdf.chunk as chunk_pdf
import rag.Llama_index_sandbox.data_ingestion_youtube.chunk as chunk_youtube
import rag.Llama_index_sandbox.embed as embed
from rag.Llama_index_sandbox import index_dir
from rag.Llama_index_sandbox.utils import timeit

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

    # NOTE: We do not index the metadata fields by video/paper link.
    #  https://docs.pinecone.io/docs/manage-indexes#selective-metadata-indexing
    #  https://docs.pinecone.io/docs/metadata-filtering
    # High cardinality consumes more memory: Pinecone indexes metadata to allow
    # for filtering. If the metadata contains many unique values — such as a unique
    # identifier for each vector — the index will consume significantly more
    # memory. Consider using selective metadata indexing to avoid indexing
    # high-cardinality metadata that is not needed for filtering.
    metadata_config = {
        "indexed": ["document_type", "title", "authors", "release_date"]
    }
    pinecone.create_index(name=index_name,
                          metadata_config=metadata_config,
                          dimension=embedding_model_chunk_size,
                          metric="cosine",
                          pod_type="p1")
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


@timeit
def create_index(embedding_model_name, embedding_model, embedding_model_chunk_size, chunk_overlap):
    logging.info("RECREATING INDEX")
    # 1. Data loading
    # pdf_links, save_dir = fetch_pdf_list(num_papers=None)
    # download_pdfs(pdf_links, save_dir)
    documents_pdfs = load_pdf.load_pdfs(directory_path=Path(pdfs_dir))  # [:1]
    documents_youtube = load_video_transcripts(directory_path=Path(video_transcripts_dir))  # [:5]

    # 2. Data chunking / text splitter
    text_chunks_pdfs, doc_idxs_pdfs = chunk_pdf.chunk_documents(documents_pdfs, chunk_size=embedding_model_chunk_size)
    text_chunks_youtube, doc_idxs_youtube = chunk_youtube.chunk_documents(documents_youtube, chunk_size=embedding_model_chunk_size)

    # 3. Manually Construct Nodes from Text Chunks
    nodes_pdf = embed.construct_node(text_chunks_pdfs, documents_pdfs, doc_idxs_pdfs)
    nodes_youtube = embed.construct_node(text_chunks_youtube, documents_youtube, doc_idxs_youtube)

    # [Optional] 4. Extract Metadata from each Node by performing LLM calls to fetch Title.
    #        We extract metadata from each Node using our Metadata extractors.
    #        This will add more metadata to each Node.
    # nodes = enrich_nodes_with_metadata_via_llm(nodes)

    # 5. Generate Embeddings for each Node
    embed.generate_embeddings(nodes_pdf, embedding_model)
    embed.generate_embeddings(nodes_youtube, embedding_model)
    nodes = nodes_pdf + nodes_youtube

    # 6. Load Nodes into a Vector Store
    # We now insert these nodes into our PineconeVectorStore.
    # NOTE: We skip the VectorStoreIndex abstraction, which is a higher-level abstraction
    # that handles ingestion as well. We use VectorStoreIndex in the next section to fast-trak retrieval/querying.
    index = load_nodes_into_vector_store_create_index(nodes, embedding_model_chunk_size)
    persist_index(index, embedding_model_name, embedding_model_chunk_size, chunk_overlap)
