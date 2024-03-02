import concurrent.futures
import logging
import random
import time

import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Union

from langchain.embeddings import OpenAIEmbeddings
from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding
from llama_index.schema import TextNode
from tiktoken.model import MODEL_TO_ENCODING

from src.Llama_index_sandbox.utils.utils import timeit
from src.Llama_index_sandbox.utils.token_counter import TokenCounter
import tiktoken

token_counter = TokenCounter(4900000)


def num_tokens_from_string(string: str, embedding_model_name: str) -> int:
    try:
        encoding = tiktoken.get_encoding(MODEL_TO_ENCODING[embedding_model_name])
        num_tokens = len(encoding.encode(string))
    except Exception as e:
        logging.error(f"Failed to get number of tokens due to: {e}")
        num_tokens = 0
        # exit(1)
    return num_tokens


def generate_node_embedding(node: TextNode, embedding_model: Union[OpenAIEmbedding, HuggingFaceEmbedding, OpenAIEmbeddings], progress_counter, total_nodes, progress_percentage=0.05):
    """Generate embedding for a single node."""
    try:
        node_content = node.get_content(metadata_mode="all")
        model_name = embedding_model.model_name if hasattr(embedding_model, 'model_name') else embedding_model.model
        if model_name in MODEL_TO_ENCODING.keys():  # deactivate token counter if it is OSS model
            num_tokens = num_tokens_from_string(node_content, model_name)

            # Add the tokens to the counter and check if the rate limit is exceeded.
            token_counter.add(num_tokens)
            if token_counter.is_rate_limit_exceeded():
                duration = random.choices([random.uniform(0, 5), random.uniform(0, 20)], weights=[0.7, 0.3])[0]
                time.sleep(duration)
                # logging.warning(f"Rate limit about to be exceeded, sleeping for {int(duration)} seconds...")

        if isinstance(embedding_model, OpenAIEmbedding) or isinstance(embedding_model, HuggingFaceEmbedding):
            node_embedding = embedding_model.get_text_embedding(node_content)
        if isinstance(embedding_model, OpenAIEmbeddings):
            node_embedding = embedding_model.embed_query(node_content)
        node.embedding = node_embedding

        with progress_counter.get_lock():
            progress_counter.value += 1
            progress = (progress_counter.value / total_nodes) * 100
            if progress_counter.value % math.ceil(total_nodes * progress_percentage) == 0 or progress_counter.value == total_nodes:
                logging.info(f"Progress: {progress:.2f}% - {progress_counter.value}/{total_nodes} nodes processed.")

    except Exception as e:
        logging.error(f"Failed to generate embedding due to: {e}")


@timeit
def generate_embeddings(nodes: List[TextNode], embedding_model):
    import concurrent.futures

    progress_counter = multiprocessing.Value('i', 0)
    total_nodes = len(nodes)

    partial_generate_node_embedding = partial(generate_node_embedding,
                                              embedding_model=embedding_model,
                                              progress_counter=progress_counter,
                                              total_nodes=total_nodes)

    num_threads = int(3/4 * multiprocessing.cpu_count())

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(executor.map(partial_generate_node_embedding, nodes))


def construct_single_node(text_chunk, src_doc_metadata):
    """Construct a single TextNode."""
    node = TextNode(text=text_chunk)
    node.metadata = src_doc_metadata
    return node


@timeit
def construct_node(text_chunks, documents, doc_idxs) -> List[TextNode]:
    """ 3. Manually Construct Nodes from Text Chunks """
    available_workers = int(1 / 2 * multiprocessing.cpu_count())  # Calculate 3/4 of the available CPU count

    with ProcessPoolExecutor(max_workers=available_workers) as executor:
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
    # logging.info(f"Sample node: {nodes[0].get_content(metadata_mode=MetadataMode.ALL)}\n\n")
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
