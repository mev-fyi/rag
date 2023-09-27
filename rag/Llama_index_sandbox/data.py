# Credits to https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
import multiprocessing
import os
import sys
# Append the parent directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import time
import random
from typing import Union, List
import requests
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index import VectorStoreIndex
from llama_index.schema import MetadataMode, TextNode
from llama_index.text_splitter import SentenceSplitter
import pandas as pd
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import logging
import math
from llama_index.embeddings import OpenAIEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.vector_stores import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
import rag.config as config
from rag.Llama_index_sandbox.utils import root_directory, RateLimitController, start_logging, timeit

root_dir = root_directory()
mev_fyi_dir = f"{root_dir}/../mev.fyi/"
research_papers_dir = f"{mev_fyi_dir}/data/paper_details.csv"
pdfs_dir = f"{mev_fyi_dir}/data/papers_pdf_downloads/"


@timeit
def fetch_pdf_list(num_papers=None):

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(research_papers_dir)

    # Append '.pdf' to the links that contain 'arxiv' and subselect all the ones which contain '.pdf'
    df['pdf_link'] = df['pdf_link'].apply(lambda link: link + '.pdf' if 'arxiv' in link else link)
    pdf_links = df.loc[df['pdf_link'].str.contains('.pdf'), 'pdf_link'].tolist()

    # If num_papers is specified, subset the list of pdf_links
    if num_papers is not None:
        pdf_links = pdf_links[:num_papers]

    # Directory to save the PDF files
    save_dir = f"{root_dir}/data/downloaded_papers"
    os.makedirs(save_dir, exist_ok=True)
    return pdf_links, save_dir, df


@timeit
def download_pdfs(pdf_links, save_dir):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(partial(download_pdf, save_dir=save_dir), pdf_links)


def download_pdf(link, save_dir):
    # Extract the file name from the link
    file_name = link.split("/")[-1]
    file_path = os.path.join(save_dir, file_name)

    # Check if the file has already been downloaded locally
    if os.path.exists(file_path):
        # logging.info(f"{file_name} already exists locally. Skipping download.")
        return

    # If not, download the file with retries
    retries = 3
    for _ in range(retries):
        try:
            logging.info(f"requesting pdf {link}")
            time.sleep(random.uniform(1, 5))
            # Send an HTTP request to the server and save the PDF file
            response = requests.get(link)
            response.raise_for_status()

            # Check if the content type is PDF
            if response.headers['Content-Type'] == 'application/pdf':
                with open(file_path, "wb") as f:
                    f.write(response.content)
                logging.info(f"Successfully downloaded {file_name}")
            else:
                logging.info(f"Failed to download a valid PDF file from {link}")

            break

        except requests.exceptions.RequestException as e:
            logging.info(f"Failed to download {link}, retrying...: {e}")
            time.sleep(0.25)  # Sleep before retrying
    else:
        logging.info(f"Failed to download {file_name} after {retries} retries.")


# Load PDFs and add metadata from the DataFrame
def load_single_pdf(paper_details_df, file_path, loader=PyMuPDFReader()):
    try:
        documents = loader.load(file_path=file_path)

        # Update 'file_path' metadata and add additional metadata
        for document in documents:
            if 'file_path' in document.metadata:
                del document.metadata['file_path']

            # Find the corresponding row in the DataFrame
            title = os.path.basename(file_path).replace('.pdf', '')
            paper_row = paper_details_df[paper_details_df['title'] == title]

            if not paper_row.empty:
                # Update metadata
                document.metadata.update({
                    'title': paper_row.iloc[0]['title'],
                    'authors': paper_row.iloc[0]['authors'],
                    'pdf_link': paper_row.iloc[0]['pdf_link'],
                    'release_date': paper_row.iloc[0]['release_date']
                })
            # TODO 2023-09-27: add relevance score as metadata. The score will be highest for research papers, ethresear.ch posts.
            #   It will be high (highest too? TBD.) for talks and conferences in YouTube video format
            #   It will be relatively lower for podcasts, tweets, and less formal content.
        return documents
    except Exception as e:
        logging.info(f"Failed to load {file_path}: {e}")
        return []


@timeit
def load_pdfs(directory_path: Union[str, Path]):
    # Convert directory_path to a Path object if it is not already
    # logging.info("Loading PDFs")
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    all_documents = []
    research_papers_path = f"{mev_fyi_dir}/data/paper_details.csv"

    paper_details_df = pd.read_csv(research_papers_path)
    partial_load_single_pdf = partial(load_single_pdf, paper_details_df=paper_details_df)
    pdf_loaded_count = 0

    # Using ThreadPoolExecutor to load PDFs in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map over all PDF files in the directory
        futures = {executor.submit(partial_load_single_pdf, file_path=pdf_file): pdf_file for pdf_file in directory_path.glob("*.pdf")}

        for future in concurrent.futures.as_completed(futures):
            pdf_file = futures[future]
            try:
                documents = future.result()
                all_documents.extend(documents)
                pdf_loaded_count += 1
            except Exception as e:
                logging.info(f"Failed to process {pdf_file}: {e}")
    logging.info(f"Successfully loaded {pdf_loaded_count} documents.")
    return all_documents


@timeit
def chunk_documents(documents, chunk_size, splitter_fn=None, chunk_overlap=None):
    # TODO 2023-09-26: We will determine if different content source better behaves with a specific text_splitter
    #  e.g. SentenceSplitter could work better for diarized YouTube videos than 'TokenTextSplitter' for instance

    if splitter_fn is None:
        splitter_fn = SentenceSplitter  # TODO 2023-09-25: The chosen text splitter should be a hyperparameter we can tune.
    if chunk_overlap is None:
        chunk_overlap = int(0.15 * chunk_size)  # TODO 2023-09-26: tune the chunk_size
    text_chunks, doc_idxs = split_text(documents, chunk_size, splitter_fn=splitter_fn, chunk_overlap=chunk_overlap, separator="\n")
    return text_chunks, doc_idxs


@timeit
def split_text(documents, chunk_size, splitter_fn, chunk_overlap, separator="\n"):
    # TODO 2023-09-25: Chunk overlap is one hyperparameter we can tune
    text_splitter = splitter_fn(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_splitter.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))
    return text_chunks, doc_idxs


def construct_single_node(text_chunk, src_doc_metadata):
    """Construct a single TextNode."""
    node = TextNode(text=text_chunk)
    node.metadata = src_doc_metadata
    return node


@timeit
def construct_node(text_chunks, documents, doc_idxs) -> List[TextNode]:
    """ 3. Manually Construct Nodes from Text Chunks """
    #  TODO 2023-09-26: should the LlamaIndex TextNode representation be scrutinized e.g. versus other implementations (e.g. Anyscale)?
    with ProcessPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(construct_single_node, text_chunks[idx], documents[doc_idxs[idx]].metadata): idx
            for idx in range(len(text_chunks))
        }

        nodes = []
        for future in concurrent.futures.as_completed(future_to_idx):
            try:
                node = future.result()
                nodes.append(node)
            except Exception as exc:
                logging.error(f"Generated an exception: {exc}")

    # print a sample node
    logging.info(f"Sample node: {nodes[0].get_content(metadata_mode=MetadataMode.ALL)}\n\n")
    return nodes


@timeit
def enrich_nodes_with_metadata_via_llm(nodes):
    """
      See part 4. Extract Metadata from each Node from https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html.
      Adds metadata to the given nodes using TitleExtractor and QuestionsAnsweredExtractor.

      The function initializes a TitleExtractor and a QuestionsAnsweredExtractor with the specified number of nodes and
      questions respectively. These extractors use a language model to generate titles and questions for the input nodes.

      Parameters:
      nodes: The input nodes to which metadata will be added.

      Returns:
      The nodes enriched with metadata.

      Note:
      This function may make multiple API calls depending on the number of nodes and questions specified,
      and the nature of the language model used.
      """
    from llama_index.node_parser.extractors import (
        MetadataExtractor,
        QuestionsAnsweredExtractor,
        TitleExtractor,
    )
    from llama_index.llms import OpenAI

    llm = OpenAI(model="gpt-3.5-turbo")

    metadata_extractor = MetadataExtractor(
        extractors=[
            TitleExtractor(nodes=5, llm=llm),
            QuestionsAnsweredExtractor(questions=3, llm=llm),  # TODO 2023-09-26: check what QuestionsAnsweredExtractor does under the hood
        ],
        in_place=False,
    )

    nodes = metadata_extractor.process_nodes(nodes)
    return nodes


def generate_node_embedding(node, embedding_model, progress_counter, total_nodes, rate_limit_controller, progress_percentage=0.05):
    """Generate embedding for a single node."""
    while True:
        try:
            node_embedding = embedding_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            if isinstance(embedding_model, OpenAIEmbedding):
                node_embedding = embedding_model.get_text_embedding(
                    node.get_content(metadata_mode="all")
                )
            elif isinstance(embedding_model, HuggingFaceEmbeddings):
                node_embedding = embedding_model.embed_documents(
                    node.get_content(metadata_mode="all")
                )
            else:
                assert False, "The embedding model is not supported."
            node.embedding = node_embedding
            with progress_counter.get_lock():
                progress_counter.value += 1
                progress = (progress_counter.value / total_nodes) * 100
                if progress_counter.value % math.ceil(total_nodes * progress_percentage) == 0 or progress_counter.value == total_nodes:
                    logging.info(f"Progress: {progress:.2f}% - {progress_counter.value}/{total_nodes} nodes processed.")
            rate_limit_controller.reset_backoff_time()
            break
        except Exception as e:
            if 'rate_limit_exceeded' in str(e):
                rate_limit_controller.register_rate_limit_exceeded()
            else:
                logging.error(f"Failed to generate embedding due to: {e}")
                break


def generate_embeddings(nodes, embedding_model):
    import concurrent.futures

    progress_counter = multiprocessing.Value('i', 0)
    total_nodes = len(nodes)
    rate_limit_controller = RateLimitController()

    partial_generate_node_embedding = partial(generate_node_embedding,
                                              embedding_model=embedding_model,
                                              progress_counter=progress_counter,
                                              total_nodes=total_nodes,
                                              rate_limit_controller=rate_limit_controller)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(partial_generate_node_embedding, nodes))


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
def load_nodes_into_vector_store_create_index(nodes, vector_store):
    """
    We now insert these nodes into our PineconeVectorStore.

    NOTE: We skip the VectorStoreIndex abstraction, which is a higher-level
    abstraction that handles ingestion as well. We use VectorStoreIndex in the next section to fast-track retrieval/querying.
    """
    vector_store.add(nodes)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index


@timeit
def retrieve_and_query_from_vector_store(index):
    # service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-0613"))
    query_engine = index.as_query_engine(similarity_top_k=5)
    # TODO 2023-09-27: if the question is totally unrelated, should the response be empty?
    query_str = "Can you tell me about the key concepts for safety finetuning"
    response = query_engine.query(query_str)
    logging.info(response)

    query_str = "Tell me about LVR"
    response = query_engine.query(query_str)
    logging.info(response)

    query_str = "What plagues current AMM designs?"
    response = query_engine.query(query_str)
    logging.info(response)

    # TODO 2023-09-27: improve the response engine with react agent chatbot.

    # chat_engine = index.as_chat_engine(chat_mode=ChatMode.REACT, verbose=True)
    # response = chat_engine.chat("Hi")
    pass


def get_embedding_model(embedding_model_name, model_kwargs=None, encode_kwargs=None):
    if embedding_model_name == "text-embedding-ada-002":
        embedding_model = OpenAIEmbedding(
            model=embedding_model_name,
            openai_api_base=os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
    else:
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},  # NOTE: using implementation default
            encode_kwargs={'normalize_embeddings': False},  # NOTE: using implementation default
            # encode_kwargs={"device": "cuda", "batch_size": 100},
        )
    return embedding_model


def run():
    start_logging()
    # 1. Data loading
    # pdf_links, save_dir = fetch_pdf_list(num_papers=None)
    # download_pdfs(pdf_links, save_dir)
    documents = load_pdfs(directory_path=Path(pdfs_dir))

    embedding_model_str = os.environ.get('EMBEDDING_MODEL_NAME_OSS')
    embedding_model_chunk_size = config.EMBEDDING_DIMENSIONS[embedding_model_str]
    embedding_model = get_embedding_model(embedding_model_name=embedding_model_str)

    # 2. Data chunking / text splitter
    text_chunks, doc_idxs = chunk_documents(documents, chunk_size=embedding_model_chunk_size)

    # 3. Manually Construct Nodes from Text Chunks
    nodes = construct_node(text_chunks, documents, doc_idxs)

    # [Optional] 4. Extract Metadata from each Node by performing LLM calls to fetch Title.
    #        We extract metadata from each Node using our Metadata extractors.
    #        This will add more metadata to each Node.
    # nodes = enrich_nodes_with_metadata_via_llm(nodes)

    # 5. Generate Embeddings for each Node
    generate_embeddings(nodes, embedding_model)

    # 6. Load Nodes into a Vector Store
    # We now insert these nodes into our PineconeVectorStore.
    # NOTE: We skip the VectorStoreIndex abstraction, which is a higher-level abstraction
    # that handles ingestion as well. We use VectorStoreIndex in the next section to fast-trak retrieval/querying.
    vector_store = initialise_vector_store(dimension=embedding_model_chunk_size)
    index = load_nodes_into_vector_store_create_index(nodes, vector_store)

    # 7. Retrieve and Query from the Vector Store
    # Now that our ingestion is complete, we can retrieve/query this vector store.
    # NOTE: We can use our high-level VectorStoreIndex abstraction here. See the next section to see how to define retrieval at a lower-level!
    retrieve_and_query_from_vector_store(index=index)

    return index
    pass
    # delete the index to save resources once we are done
    # vector_store.delete(deleteAll=True)


if __name__ == "__main__":
    run()
