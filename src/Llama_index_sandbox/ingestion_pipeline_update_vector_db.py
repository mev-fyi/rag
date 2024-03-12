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
        # docstore_strategy=DocstoreStrategy.UPSERTS,
        docstore=SimpleDocumentStore(),
    )
    if os.path.exists(f"{root_directory()}/pipeline_storage"):
        try:
            pipeline.load(persist_dir=f"{root_directory()}/pipeline_storage")
        except Exception as e:
            logging.error(f"Failed to load pipeline from storage with error [{e}], continuing with new pipeline.")

    return pipeline


@timeit
def create_index(add_new_transcripts=False, num_files=None):
    copy_and_verify_files()
    logging.info("Starting Index Creation Process")
    # TODO 2024-03-06: need to locally store the embeddings. That requires going offline since with the pipeline we first delete the index
    #  then do the embedding and in the same pipeline method, insert the vectors, and only then we persist to the docstore.
    #  Likewise we can't store the embedding, double it up to the vector index, delete the index, load the pipeline, and re-store the vectors.
    #  Or can we? with two instances? Or as follows:
    #   1. Instantiate existing pinecone vector
    #   2. Embed
    #   3. Insert vectors (doubled since index not deleted)
    #   4. Load up another pipeline object where we delete the index, and insert the vectors.
    #   5. BUT the problem is that loading back, does not yield the Documents object back, it just "loads" the pipeline object.
    #   6. While running the pipeline and performing the inserts requires having the Documents objects.

    # I can simply:
    #   1. instantiate without vector store
    #   2. run the pipeline
    #   3. save to docstore
    #   4. instantiate a new pipeline, delete the vector store, add the vector store, and add the nodes of the first pipeline to the vector store of the second pipeline

    overwrite = True  # whether we overwrite DB namely we load all documents instead of only loading the increment since last database update
    num_files = 1
    files_window = None# (20, 100)

    # TODO 2024-03-12: the whole process can be largely improved if we batched process end to end the docs to inserting the vectors

    # STEP 1: load all docs
    documents_pdfs = load_pdf.load_pdfs(directory_path=Path(PDF_DIRECTORY), num_files=0, files_window=files_window, overwrite=overwrite)
    documents_pdfs += load_docs.load_docs_as_pdf(num_files=0, files_window=files_window, overwrite=overwrite)
    documents_pdfs += load_articles.load_pdfs(directory_path=Path(ARTICLES_DIRECTORY), num_files=0, files_window=files_window, overwrite=overwrite)
    documents_pdfs += load_discourse_articles.load_pdfs(directory_path=Path(DISCOURSE_ARTICLES_DIRECTORY), num_files=num_files, files_window=files_window, overwrite=overwrite)
    documents_youtube = load_video_transcripts(directory_path=Path(YOUTUBE_VIDEO_DIRECTORY), add_new_transcripts=add_new_transcripts, num_files=None, files_window=files_window, overwrite=overwrite)

    # STEP 2: EMBED ALL DOCS + SAVE TO DOCSTORE LOCALLY WITHOUT VECTOR STORE (TO BE USED IN STEP 4)
    pipeline_v1 = initialise_pipeline(add_to_vector_store=True)
    nodes = pipeline_v1.run(documents=documents_pdfs + documents_youtube, num_workers=15, show_progress=True)
    if len(nodes) > 0:
        pipeline_v1.persist(persist_dir=f"{root_directory()}/pipeline_storage")
    logging.info(f"Processed {len(nodes)} Nodes")

    # STEP 3: SPIN OFF NEW PIPELINE INSTANCE WITH VECTOR STORE WHICH IS DELETED AND CREATED BACK
    pipeline_v2 = initialise_pipeline()
    # STEP 4: ADD EMBEDDINGS TO VECTOR STORE
    if pipeline_v2.vector_store is not None:
        nodes_to_add = [n for n in nodes if n.embedding is not None]
        pipeline_v2.vector_store.add(nodes_to_add)
        logging.info(f"Added {len(nodes_to_add)} Nodes to vector database")

    logging.info("Index Creation Process Completed")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    create_index()
