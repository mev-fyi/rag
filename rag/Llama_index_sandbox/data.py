import os
import time
import random
from typing import Union, List

import requests
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
import llama_index
from llama_index.chat_engine.types import ChatMode
from llama_index.llms import OpenAI
import pandas as pd
import concurrent.futures
from functools import partial
from pathlib import Path
import logging

from llama_index.schema import MetadataMode

from rag.Llama_index_sandbox.utils import root_directory

# https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html


def fetch_pdf_list():
    root_dir = root_directory()
    mev_fyi_dir = f"{root_dir}/../mev.fyi/"
    research_papers_path = f"{mev_fyi_dir}/data/paper_details.csv"

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(research_papers_path)

    # Append '.pdf' to the links that contain 'arxiv' and subselect all the ones which contain '.pdf'
    df['pdf_link'] = df['pdf_link'].apply(lambda link: link + '.pdf' if 'arxiv' in link else link)
    pdf_links = df.loc[df['pdf_link'].str.contains('.pdf'), 'pdf_link'].tolist()
    # logging.info(f"\nThis is pdf_links: {pdf_links}\n")

    # Directory to save the PDF files
    save_dir = "downloaded_papers"
    os.makedirs(save_dir, exist_ok=True)
    return pdf_links, save_dir


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
            # Send a HTTP request to the server and save the PDF file
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


def load_single_pdf(file_path, loader=PyMuPDFReader()):
    try:
        return loader.load(file_path=file_path)
    except Exception as e:
        logging.info(f"Failed to load {file_path}: {e}")
        return []


def load_pdfs(directory_path: Union[str, Path]):
    # Convert directory_path to a Path object if it is not already
    logging.info("Loading PDFs")
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    all_documents = []

    # Using ThreadPoolExecutor to load PDFs in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map over all PDF files in the directory
        futures = {executor.submit(load_single_pdf, pdf_file): pdf_file for pdf_file in directory_path.glob("*.pdf")}

        for future in concurrent.futures.as_completed(futures):
            pdf_file = futures[future]
            try:
                documents = future.result()
                all_documents.extend(documents)
            except Exception as e:
                logging.info(f"Failed to process {pdf_file}: {e}")
    return all_documents


def split_text(documents, chunk_size, chunk_overlap=None, separator="\n"):
    # TODO 2023-09-25: The chosen text splitter should be a hyperparameter we can tune
    from llama_index.text_splitter import SentenceSplitter

    # TODO 2023-09-25: Chunk overlap is one hyperparameter we can tune
    if chunk_overlap is None:
        chunk_overlap = int(0.15 * chunk_size)
    text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_splitter.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))
    return text_chunks, doc_idxs


def construct_node(text_chunks, documents, doc_idxs) -> List[llama_index.schema.TextNode]:
    """ 3. Manually Construct Nodes from Text Chunks """
    from llama_index.schema import TextNode

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    # print a sample node
    logging.info(f"Sample node: {nodes[0].get_content(metadata_mode=MetadataMode.ALL)}")
    return nodes


def add_metadata_to_node(nodes):
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
            QuestionsAnsweredExtractor(questions=3, llm=llm),
        ],
        in_place=False,
    )

    nodes = metadata_extractor.process_nodes(nodes)
    return nodes


def generate_embeddings(nodes, embed_model=None) -> None:
    """
    Generates and assigns embeddings to the provided nodes.

    The function utilizes OpenAIEmbedding (or another Embedding model) to generate embeddings
    for each node based on its content with metadata_mode set to "all". # TODO 2023-09-25 highlight what metadata_mode set to "all" means
    The generated embedding is then assigned to the embedding attribute of the node.

    Parameters:
    nodes (list): A list of nodes for which embeddings are to be generated and assigned.

    Returns:
    None: The function modifies the input nodes in-place and does not return anything.
    """
    from llama_index.embeddings import OpenAIEmbedding
    # TODO 2023-09-25 use Anyscale upgrade to smoothly use other embedding models

    if embed_model is None:
        embed_model = OpenAIEmbedding()

    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding


def initialise_vector_store():
    # TODO 2023-09-25: upgrade for parametrisable vector_store
    import pinecone
    api_key = os.environ["PINECONE_API_KEY"]
    pinecone.init(api_key=api_key, environment="us-west1-gcp")

    # TODO 2023-09-25: optimise for distance metric here

    # dimensions are for text-embedding-ada-002
    pinecone.create_index("quickstart", dimension=1536, metric="euclidean", pod_type="p1")
    pinecone_index = pinecone.Index("quickstart")
    # [Optional] drop contents in index
    pinecone_index.delete(deleteAll=True)
    return pinecone_index


def load_nodes_into_vector_store(nodes, vector_store):
    """
    We now insert these nodes into our PineconeVectorStore.

    NOTE: We skip the VectorStoreIndex abstraction, which is a higher-level
    abstraction that handles ingestion as well. We use VectorStoreIndex in the next section to fast-trak retrieval/querying.
    """
    vector_store.add(nodes)


def run():
    # 1. Data loading
    pdf_links, save_dir = fetch_pdf_list()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(partial(download_pdf, save_dir=save_dir), pdf_links)
    documents = load_pdfs(directory_path=Path(save_dir))

    # 2. Data chunking / text splitter
    text_chunks, doc_idxs = split_text(documents, chunk_size=1024)

    # 3. Manually Construct Nodes from Text Chunks
    nodes = construct_node(text_chunks, documents, doc_idxs)

    # 4. Extract Metadata from each Node
    #        We extract metadata from each Node using our Metadata extractors.
    #        This will add more metadata to each Node.
    nodes = add_metadata_to_node(nodes)

    # 5. Generate Embeddings for each Node
    generate_embeddings(nodes)

    # 6. Load Nodes into a Vector Store
    # We now insert these nodes into our PineconeVectorStore.
    # NOTE: We skip the VectorStoreIndex abstraction, which is a higher-level abstraction
    # that handles ingestion as well. We use VectorStoreIndex in the next section to fast-trak retrieval/querying.
    vector_store = initialise_vector_store()
    load_nodes_into_vector_store(nodes, vector_store)

    # Data indexing
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-0613"))
    index = VectorStoreIndex.from_documents(data, service_context=service_context)
    chat_engine = index.as_chat_engine(chat_mode=ChatMode.REACT, verbose=True)
    response = chat_engine.chat("Hi")
    logging.info(response)


if __name__ == "__main__":
    logging.info("\nSCRIPT STARTING\n"*2)
    run()
