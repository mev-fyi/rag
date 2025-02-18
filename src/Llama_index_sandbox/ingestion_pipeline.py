import logging
import os
from pathlib import Path
import time

from llama_index.embeddings.openai import OpenAIEmbedding

from src.Llama_index_sandbox import PDF_DIRECTORY, YOUTUBE_VIDEO_DIRECTORY, ARTICLES_DIRECTORY, DISCOURSE_ARTICLES_DIRECTORY
import src.Llama_index_sandbox.data_ingestion_pdf.load as load_pdf
import src.Llama_index_sandbox.data_ingestion_pdf.load_articles as load_articles
from src.Llama_index_sandbox import config_instance
from llama_index.core.storage.docstore import SimpleDocumentStore
from src.Llama_index_sandbox.data_ingestion_pdf import load_discourse_articles, load_docs, load_ethglobal_docs
from src.Llama_index_sandbox.data_ingestion_youtube.load.load import load_video_transcripts
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
)
from llama_index.core.node_parser import SentenceSplitter
from src.Llama_index_sandbox.utils.utils import timeit, root_directory, copy_and_verify_files, load_vector_store_from_pinecone_database


def initialise_pipeline(add_to_vector_store=True, delete_old_index=False, new_index=False, index_name="mevfyi-cosine"):
    if add_to_vector_store:
        vector_store = load_vector_store_from_pinecone_database(delete_old_index=delete_old_index, new_index=new_index, index_name=index_name)
    else:
        vector_store = None

    embedding_model = OpenAIEmbedding()

    if new_index:
        docstore_strategy = DocstoreStrategy.UPSERTS
    else:
        docstore_strategy = DocstoreStrategy.DUPLICATES_ONLY

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=config_instance.CHUNK_SIZES[0], chunk_overlap=config_instance.CHUNK_OVERLAPS[0]),
            embedding_model,
        ],
        vector_store=vector_store,
        docstore_strategy=docstore_strategy,
        docstore=SimpleDocumentStore(),
    )
    if os.path.exists(f"{root_directory()}/pipeline_storage") and not new_index:
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


def calculate_batch_size(total_docs):
    """
    Calculate the batch size dynamically based on the number of documents.
    The batch size scales with the number of documents to ensure efficient processing
    while avoiding too large batches that risk substantial data loss on failure.

    :param total_docs: Total number of documents to be processed
    :return: An appropriate batch size
    """
    logging.info(f"[calculate_batch_size] Calculating batch size for [{total_docs}] documents")
    if total_docs < 100:
        return max(1, total_docs // 10)  # Small number of documents, process in tens
    elif total_docs < 500:
        return max(1, total_docs // 50)  # Medium number, process in twenties
    elif total_docs < 1000:
        return max(1, total_docs // 100)  # Larger set, process in fifties
    else:
        return max(1, total_docs // 200)  # Very large set, process in hundreds


@timeit
def create_index(add_new_transcripts=True, num_files=None):
    copy_and_verify_files()
    copy_docstore()
    logging.info("Starting Index Load Process")

    overwrite = False  # whether we overwrite DB namely we load all documents instead of only loading the increment since last database update
    num_files = None
    files_window = None  # (20, 100)
    add_new_transcripts = True

    # Load all docs
    documents_pdfs = []
    config_names = None  # ['chainlink_data_feeds', 'chainlink_vrf']
    documents_pdfs += load_ethglobal_docs.load_docs_as_pdf(debug=True, num_files=num_files, files_window=files_window, overwrite=overwrite, config_names=config_names)
    documents_pdfs += load_docs.load_docs_as_pdf(num_files=num_files, files_window=files_window, overwrite=overwrite)
    documents_pdfs += load_pdf.load_pdfs(directory_path=Path(PDF_DIRECTORY), num_files=num_files, files_window=files_window, overwrite=overwrite)
    documents_pdfs += load_articles.load_pdfs(directory_path=Path(ARTICLES_DIRECTORY), num_files=num_files, files_window=files_window, overwrite=overwrite)
    documents_pdfs += load_discourse_articles.load_pdfs(directory_path=Path(DISCOURSE_ARTICLES_DIRECTORY), num_files=num_files, files_window=files_window, overwrite=overwrite)
    documents_youtube = []
    documents_youtube += load_video_transcripts(directory_path=Path(YOUTUBE_VIDEO_DIRECTORY), add_new_transcripts=add_new_transcripts, num_files=None, files_window=files_window, overwrite=overwrite)

    all_documents = documents_pdfs + documents_youtube
    total_docs = len(all_documents)
    batch_size = calculate_batch_size(total_docs)  # Calculate dynamic batch size based on document count

    pipeline = initialise_pipeline(add_to_vector_store=True)
    all_nodes = []

    # Process documents in batches with retries
    for i in range(0, total_docs, batch_size):
        batch_documents = all_documents[i:i + batch_size]
        logging.info(f"Processing batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size} Nodes of length {len(batch_documents)}")

        retries = 0
        success = False
        while not success and retries < 5:  # Retry up to 5 times
            try:
                nodes = pipeline.run(documents=batch_documents, num_workers=18, show_progress=True)
                all_nodes.extend(nodes)
                if len(nodes) > 0:
                    pipeline.persist(persist_dir=f"{root_directory()}/pipeline_storage")
                success = True
            except Exception as e:
                retries += 1
                wait_time = 2 ** retries  # Exponential backoff
                logging.error(f"Failed to process batch {i // batch_size + 1}, retrying in {wait_time} seconds. Error: {e}")
                time.sleep(wait_time)

        logging.info(f"Processed batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size} Nodes")

    logging.info(f"Processed {len(all_nodes)} Nodes in total")
    logging.info("Index Load Process Completed")


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    create_index()


if __name__ == "__main__":
    main()
