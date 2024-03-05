import logging
import multiprocessing
import os
from typing import List
from pathlib import Path
from src.Llama_index_sandbox import PDF_DIRECTORY, YOUTUBE_VIDEO_DIRECTORY, config_instance, ARTICLES_DIRECTORY, DISCOURSE_ARTICLES_DIRECTORY
import src.Llama_index_sandbox.data_ingestion_pdf.load as load_pdf
import src.Llama_index_sandbox.data_ingestion_pdf.load_articles as load_articles
import src.Llama_index_sandbox.data_ingestion_pdf.chunk as chunk_pdf
import src.Llama_index_sandbox.data_ingestion_youtube.chunk as chunk_youtube
from src.Llama_index_sandbox import config_instance
from src.Llama_index_sandbox.data_ingestion_pdf import load_discourse_articles, load_docs
from src.Llama_index_sandbox.data_ingestion_youtube.load.load import load_video_transcripts
from src.Llama_index_sandbox.embed import generate_embeddings
from src.Llama_index_sandbox.data_ingestion_pdf.chunk import chunk_documents as chunk_pdf_documents
from src.Llama_index_sandbox.data_ingestion_youtube.chunk import chunk_documents as chunk_youtube_documents
import src.Llama_index_sandbox.embed as embed
from multiprocessing import Pool

from src.Llama_index_sandbox.index import initialise_vector_store
from src.Llama_index_sandbox.utils.utils import timeit, get_embedding_model


def process_nodes_subset(nodes_subset, embedding_model, doc_type):
    """
    Process a subset of nodes: chunk, embed, and optionally enrich with metadata.
    Args:
        nodes_subset: A list of documents to be processed.
        embedding_model: The embedding model to generate embeddings.
        doc_type: Type of document, e.g., 'pdf' or 'youtube'.
    Returns:
        Processed nodes ready for insertion into vector store.
    """
    text_splitter_chunk_size = config_instance.CHUNK_SIZES[0]
    text_splitter_chunk_overlap_percentage = config_instance.CHUNK_OVERLAPS[0]
    # Depending on the document type, use the appropriate chunking function
    if doc_type == 'pdf':
        text_chunks, doc_idxs = chunk_pdf_documents(nodes_subset, text_splitter_chunk_size, text_splitter_chunk_overlap_percentage)
    elif doc_type == 'youtube':
        text_chunks, doc_idxs = chunk_youtube_documents(nodes_subset, text_splitter_chunk_size, text_splitter_chunk_overlap_percentage)
    else:
        raise ValueError("Unsupported document type.")

    nodes = embed.construct_node(text_chunks, nodes_subset, doc_idxs)
    generate_embeddings(nodes, embedding_model)
    return nodes


@timeit
def parallel_ingestion(documents, embedding_model, doc_type, num_workers=4):
    """
    Ingest documents into the vector store in parallel.
    Args:
        documents: List of documents to ingest.
        embedding_model: Embedding model for generating embeddings.
        doc_type: Type of documents in the list.
        num_workers: Number of parallel processes to use.
    """
    # Split documents into subsets for each worker
    subset_size = len(documents) // num_workers
    documents_subsets = [documents[i:i + subset_size] for i in range(0, len(documents), subset_size)]

    # Process each subset in parallel
    with Pool(num_workers) as pool:
        results = pool.starmap(process_nodes_subset, [(subset, embedding_model, doc_type) for subset in documents_subsets])

    # Flatten the list of nodes from all processes
    nodes = [node for subset in results for node in subset]
    return nodes


@timeit
def parallel_process_documents(documents, embedding_model, doc_type, num_workers=4):
    # Adaptation of the parallel_ingestion function to incorporate the specific document processing functions
    # Use multiprocessing to distribute document processing across multiple workers
    subset_size = max(1, len(documents) // num_workers)  # Ensure at least one document per subset
    documents_subsets = [documents[i:i + subset_size] for i in range(0, len(documents), subset_size)]

    with Pool(num_workers) as pool:
        results = pool.starmap(process_documents_subset, [(subset, embedding_model, doc_type) for subset in documents_subsets])

    nodes = [node for subset in results for node in subset]
    return nodes


def initialise_pipeline():
    from llama_index.core.ingestion import (
        DocstoreStrategy,
        IngestionPipeline,
        IngestionCache,
    )
    # from llama_index.core.ingestion.cache import RedisCache
    from llama_index.vector_stores.redis import RedisVectorStore
    from llama_index.storage.docstore.redis import RedisDocumentStore
    from llama_index.core.node_parser import SentenceSplitter

    vector_store = initialise_vector_store(embedding_model_vector_dimension=config_instance.EMBEDDING_DIMENSIONS[config_instance.EMBEDDING_MODELS[0]],
                                           vector_space_distance_metric=config_instance.VECTOR_SPACE_DISTANCE_METRIC[0])

    # cache = IngestionCache(
    #     cache=RedisCache.from_host_and_port("localhost", 6379),
    #     collection="redis_cache",
    # )

    # Optional: clear vector store if exists
    # if vector_store._index_exists():
    #     vector_store.delete_index()

    embedding_model = get_embedding_model(embedding_model_name=config_instance.EMBEDDING_MODELS[0])

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=config_instance.CHUNK_SIZES[0], chunk_overlap=config_instance.CHUNK_OVERLAPS[0]),
            embedding_model,
        ],
        # docstore=RedisDocumentStore.from_host_and_port(
        #     "localhost", 6379, namespace="document_store"
        # ),
        vector_store=vector_store,
        # cache=cache,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    return pipeline


@timeit
def create_index(model_details, embedding_model, add_new_transcripts, vector_space_distance_metric, num_files=None):
    logging.info("Starting Index Creation Process")

    documents_pdfs = load_pdf.load_pdfs(directory_path=Path(PDF_DIRECTORY), num_files=num_files)
    documents_pdfs += load_articles.load_pdfs(directory_path=Path(ARTICLES_DIRECTORY), num_files=num_files)
    documents_pdfs += load_discourse_articles.load_pdfs(directory_path=Path(DISCOURSE_ARTICLES_DIRECTORY), num_files=num_files)
    documents_pdfs += load_docs.load_docs_as_pdf(num_files=num_files)
    documents_youtube = load_video_transcripts(directory_path=Path(YOUTUBE_VIDEO_DIRECTORY), add_new_transcripts=add_new_transcripts, num_files=num_files)

    # nodes_pdf = parallel_process_documents(documents_pdfs, embedding_model, 'pdf', num_workers=config_instance.PARALLEL_WORKERS)
    # nodes_youtube = parallel_process_documents(documents_youtube, embedding_model, 'youtube', num_workers=config_instance.PARALLEL_WORKERS)
    # vector_store = initialise_vector_store(embedding_model_vector_dimension=config_instance.EMBEDDING_DIMENSIONS[model_details[0]], vector_space_distance_metric=vector_space_distance_metric)

    pipeline = initialise_pipeline()
    pipeline.run(documents=documents_pdfs + documents_youtube, num_workers=int(os.cpu_count())-3)
    pipeline.persist("./pipeline_storage")

    # Assuming a function to add nodes to vector store, replace `vector_store.add` with the appropriate method call if different
    # vector_store.add(nodes_pdf + nodes_youtube)

    index = CustomVectorStoreIndex.from_vector_store(vector_store, None)  # Replace None with actual service context if needed
    persist_index(index)

    nodes = pipeline.run(documents=docs)

    logging.info("Index Creation Process Completed")