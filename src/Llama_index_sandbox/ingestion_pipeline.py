import logging
import os
from pathlib import Path

from llama_index.embeddings.openai import OpenAIEmbedding

from src.Llama_index_sandbox import PDF_DIRECTORY, YOUTUBE_VIDEO_DIRECTORY, ARTICLES_DIRECTORY, DISCOURSE_ARTICLES_DIRECTORY
import src.Llama_index_sandbox.data_ingestion_pdf.load as load_pdf
import src.Llama_index_sandbox.data_ingestion_pdf.load_articles as load_articles
from src.Llama_index_sandbox import config_instance
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from src.Llama_index_sandbox.data_ingestion_pdf import load_discourse_articles, load_docs
from src.Llama_index_sandbox.data_ingestion_youtube.load.load import load_video_transcripts
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
)
from llama_index.core.node_parser import SentenceSplitter
from src.Llama_index_sandbox.utils.utils import timeit, get_embedding_model, root_directory


def load_vector_store_from_pinecone_database():
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
    index_name = "mevfyi"
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    return vector_store


def initialise_pipeline():
    vector_store = load_vector_store_from_pinecone_database()

    embedding_model = OpenAIEmbedding()

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=config_instance.CHUNK_SIZES[0], chunk_overlap=config_instance.CHUNK_OVERLAPS[0]),
            embedding_model,
        ],
        vector_store=vector_store,
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
        docstore=SimpleDocumentStore(),
    )
    if os.path.exists(f"{root_directory()}/pipeline_storage"):
        try:
            pipeline.load(persist_dir=f"{root_directory()}/pipeline_storage")
        except Exception as e:
            logging.error(f"Failed to load pipeline from storage with error [{e}], continuing with new pipeline.")

    return pipeline


@timeit
def create_index(add_new_transcripts=False, num_files=10):
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

    overwrite = False
    documents_pdfs = load_pdf.load_pdfs(directory_path=Path(PDF_DIRECTORY), num_files=num_files, overwrite=overwrite)
    documents_pdfs += load_articles.load_pdfs(directory_path=Path(ARTICLES_DIRECTORY), num_files=num_files, overwrite=overwrite)
    documents_pdfs += load_discourse_articles.load_pdfs(directory_path=Path(DISCOURSE_ARTICLES_DIRECTORY), num_files=num_files, overwrite=overwrite)
    documents_pdfs += load_docs.load_docs_as_pdf(num_files=num_files, overwrite=overwrite)
    documents_youtube = load_video_transcripts(directory_path=Path(YOUTUBE_VIDEO_DIRECTORY), add_new_transcripts=add_new_transcripts, num_files=num_files, overwrite=overwrite)

    pipeline = initialise_pipeline()
    nodes = pipeline.run(documents=documents_pdfs + documents_youtube, num_workers=int(os.cpu_count())-3, show_progress=True)
    logging.info(f"Ingested {len(nodes)} Nodes")
    pipeline.persist(persist_dir=f"{root_directory()}/pipeline_storage")

    logging.info("Index Creation Process Completed")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    create_index()
