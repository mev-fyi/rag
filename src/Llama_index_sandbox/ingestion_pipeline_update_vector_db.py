import logging
import os
from pathlib import Path

from llama_index.embeddings.openai import OpenAIEmbedding

from src.Llama_index_sandbox import PDF_DIRECTORY, YOUTUBE_VIDEO_DIRECTORY, ARTICLES_DIRECTORY, DISCOURSE_ARTICLES_DIRECTORY
import src.Llama_index_sandbox.data_ingestion_pdf.load as load_pdf
import src.Llama_index_sandbox.data_ingestion_pdf.load_articles as load_articles
from src.Llama_index_sandbox import config_instance
from llama_index.core.storage.docstore import SimpleDocumentStore
from src.Llama_index_sandbox.data_ingestion_pdf import load_discourse_articles, load_docs
from src.Llama_index_sandbox.data_ingestion_youtube.load.load import load_video_transcripts
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
)
from llama_index.core.node_parser import SentenceSplitter
from src.Llama_index_sandbox.utils.utils import timeit, root_directory, copy_and_verify_files, load_vector_store_from_pinecone_database


def initialise_pipeline(add_to_vector_store=True, delete_old_index=False):
    if add_to_vector_store:
        vector_store = load_vector_store_from_pinecone_database(delete_old_index=delete_old_index)
    else:
        vector_store = None

    embedding_model = OpenAIEmbedding()

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=config_instance.CHUNK_SIZES[0], chunk_overlap=config_instance.CHUNK_OVERLAPS[0]),
            embedding_model,
        ],
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
        # docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
        # docstore_strategy=DocstoreStrategy.UPSERTS,
        docstore=SimpleDocumentStore(),
    )
    if os.path.exists(f"{root_directory()}/pipeline_storage"):
        try:
            pipeline.load(persist_dir=f"{root_directory()}/pipeline_storage")
        except Exception as e:
            logging.error(f"Failed to load pipeline from storage with error [{e}], continuing with new pipeline.")

    return pipeline


def copy_docstore():
    import shutil
    from datetime import datetime
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Define the source and destination paths
    source_path = f"{root_directory()}/pipeline_storage/docstore.json"
    destination_dir = f"{root_directory()}/temp/"
    destination_path = os.path.join(destination_dir, f"docstore_{timestamp}.json")

    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Copy the file
    shutil.copy(source_path, destination_path)
    logging.info(f"Docstore backup created at {destination_path}")


@timeit
def create_index(add_new_transcripts=False, num_files=None):
    copy_and_verify_files()
    copy_docstore()
    logging.info("Starting Index Creation Process")

    overwrite = False  # whether we overwrite DB namely we load all documents instead of only loading the increment since last database update
    num_files = 1
    files_window = None  # (20, 100)

    # Load all docs
    documents_pdfs = load_pdf.load_pdfs(directory_path=Path(PDF_DIRECTORY), num_files=0, files_window=files_window, overwrite=overwrite)
    documents_pdfs += load_docs.load_docs_as_pdf(num_files=0, files_window=files_window, overwrite=overwrite)
    documents_pdfs += load_articles.load_pdfs(directory_path=Path(ARTICLES_DIRECTORY), num_files=0, files_window=files_window, overwrite=overwrite)
    documents_pdfs += load_discourse_articles.load_pdfs(directory_path=Path(DISCOURSE_ARTICLES_DIRECTORY), num_files=1, files_window=files_window, overwrite=overwrite)
    documents_youtube = load_video_transcripts(directory_path=Path(YOUTUBE_VIDEO_DIRECTORY), add_new_transcripts=add_new_transcripts, num_files=None, files_window=files_window, overwrite=overwrite)

    all_documents = documents_pdfs + documents_youtube
    total_docs = len(all_documents)
    batch_size = max(1, total_docs // 100)  # Ensure batch_size is at least 1

    pipeline = initialise_pipeline(add_to_vector_store=True)
    all_nodes = []

    # Process documents in batches
    for i in range(0, total_docs, batch_size):
        batch_documents = all_documents[i:i+batch_size]
        nodes = pipeline.run(documents=batch_documents, num_workers=18, show_progress=True)
        all_nodes.extend(nodes)
        if len(nodes) > 0:
            pipeline.persist(persist_dir=f"{root_directory()}/pipeline_storage")
        logging.info(f"Processed batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} Nodes")

    logging.info(f"Processed {len(all_nodes)} Nodes in total")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    create_index()
