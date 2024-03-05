import logging
import os
from pathlib import Path
from src.Llama_index_sandbox import PDF_DIRECTORY, YOUTUBE_VIDEO_DIRECTORY, config_instance, ARTICLES_DIRECTORY, DISCOURSE_ARTICLES_DIRECTORY
import src.Llama_index_sandbox.data_ingestion_pdf.load as load_pdf
import src.Llama_index_sandbox.data_ingestion_pdf.load_articles as load_articles
from src.Llama_index_sandbox import config_instance
from src.Llama_index_sandbox.data_ingestion_pdf import load_discourse_articles, load_docs
from src.Llama_index_sandbox.data_ingestion_youtube.load.load import load_video_transcripts

from src.Llama_index_sandbox.index import initialise_vector_store
from src.Llama_index_sandbox.utils.utils import timeit, get_embedding_model


def initialise_pipeline():
    from llama_index.core.ingestion import (
        DocstoreStrategy,
        IngestionPipeline,
    )
    from llama_index.core.node_parser import SentenceSplitter

    vector_store = initialise_vector_store(embedding_model_vector_dimension=config_instance.EMBEDDING_DIMENSIONS[config_instance.EMBEDDING_MODELS[0]],
                                           vector_space_distance_metric=config_instance.VECTOR_SPACE_DISTANCE_METRIC[0])

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