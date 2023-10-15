import concurrent.futures
import logging
import random
import time

import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List

from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding
from llama_index.schema import TextNode, MetadataMode

from rag.Llama_index_sandbox.utils import RateLimitController, timeit


def generate_node_embedding(node: TextNode, embedding_model: OpenAIEmbedding, progress_counter, total_nodes, rate_limit_controller, progress_percentage=0.05):
    """Generate embedding for a single node."""
    while True:
        try:
            # random sleep from 0 to 3 seconds
            time.sleep(random.uniform(0, 2))
            node_embedding = embedding_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding
            with progress_counter.get_lock():
                progress_counter.value += 1
                progress = (progress_counter.value / total_nodes) * 100
                if progress_counter.value % math.ceil(total_nodes * progress_percentage) == 0 or progress_counter.value == total_nodes:
                    logging.info(f"Progress: {progress:.2f}% - {progress_counter.value}/{total_nodes} nodes processed.")
            rate_limit_controller.reset_backoff_time()
            break
        except Exception as e:
            if 'Rate limit reached' in str(e) or 'rate_limit_exceeded' in str(e):
                logging.warning("Rate limit error detected.")
                rate_limit_controller.register_rate_limit_exceeded()
            else:
                logging.error(f"Failed to generate embedding due to: {e}")
                break


def generate_embeddings(nodes: List[TextNode], embedding_model):
    import concurrent.futures

    progress_counter = multiprocessing.Value('i', 0)
    total_nodes = len(nodes)
    rate_limit_controller = RateLimitController()

    partial_generate_node_embedding = partial(generate_node_embedding,
                                              embedding_model=embedding_model,
                                              progress_counter=progress_counter,
                                              total_nodes=total_nodes,
                                              rate_limit_controller=rate_limit_controller)

    num_threads = multiprocessing.cpu_count()  # half the number of CPUs

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(executor.map(partial_generate_node_embedding, nodes))


def get_embedding_model(embedding_model_name):
    if embedding_model_name == "text-embedding-ada-002":
        embedding_model = OpenAIEmbedding()
    else:
        embedding_model = HuggingFaceEmbedding(
            tokenizer_name=embedding_model_name
        )
    # else:
    #     assert False, f"The embedding model is not supported: [{embedding_model_name}]"
    return embedding_model


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
