# Credits to https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html

import os
from pathlib import Path
import logging

from llama_index.agent import ReActAgent
from llama_index.chat_engine import CondenseQuestionChatEngine

import rag.config as config
from rag.Llama_index_sandbox.utils import start_logging, timeit
from rag.Llama_index_sandbox import pdfs_dir
from rag.Llama_index_sandbox.data_ingestion_pdf.load import load_pdfs, fetch_pdf_list, download_pdfs
from rag.Llama_index_sandbox.data_ingestion_pdf.chunk import chunk_documents, get_chunk_overlap
from rag.Llama_index_sandbox.data_ingestion_pdf.embed import generate_embeddings, get_embedding_model, construct_node
from rag.Llama_index_sandbox.data_ingestion_pdf.index import initialise_vector_store, load_nodes_into_vector_store_create_index, load_index_from_disk, persist_index
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from rag.Llama_index_sandbox.store_response import store_response
from functools import partial
from rag.Llama_index_sandbox.constants import SYSTEM_MESSAGE


def log_and_store(store_response_fn, query_str, response):
    logging.info(response)
    # store_response_fn(query_str, response)


def get_chat_engine(index, service_context, chat_mode="react", verbose=True, similarity_top_k=5):
    from llama_index.prompts import PromptTemplate
    from llama_index.llms import ChatMessage, MessageRole

    # TODO 2023-09-29: make sure the temperature is set to zero for consistent results
    # TODO 2023-09-29: creating a (react) chat engine from an index transforms that
    #  query as a tool and passes it to the agent under the hood. That query tool can receive a description.
    #  We need to determine (1) if we pass several query engines as tool or build a massive single one (cost TBD),
    #  and (2) if we pass a description to the query tool and what is the expected retrieval impact from having a description versus not.
    # TODO 2023-09-29: use lower-level construction than as_chat_engine()

    # TODO 2023-09-29: add system prompt to agent. BUT it is an input to OpenAI agent but not React Agent!
    #   OpenAI agent has prefix_messages in its constructor, but React Agent does not.
    chat_engine = index.as_chat_engine(chat_mode=chat_mode,
                                       verbose=verbose,
                                       similarity_top_k=similarity_top_k,
                                       service_context=service_context,
                                       system_prompt=SYSTEM_MESSAGE)
    return chat_engine


@timeit
def retrieve_and_query_from_vector_store(embedding_model_name, llm_model_name, chunksize, chunkoverlap, index):
    # TODO 2023-09-29: we need to set in stone an accurate baseline evaluation using ReAct agent.
    #   To achieve this we need to save intermediary Response objects to make sure we can distill results and have access to nodes and chunks used for the reasoning
    # TODO 2023-09-29: determine how we should structure our indexes per document type
    service_context = ServiceContext.from_defaults(llm=OpenAI(model=llm_model_name))
    # create partial store_response with everything but the query_str and response
    store_response_partial = partial(store_response, embedding_model_name, llm_model_name, chunksize, chunkoverlap)

    # chat_engine = index.as_chat_engine(chat_mode="react", verbose=True, similarity_top_k=5, service_context=service_context)
    chat_engine = get_chat_engine(index, service_context, chat_mode="react", verbose=True, similarity_top_k=5)

    queries = [
        "What is red teaming in AI",  # Should refuse to respond,
        "Tell me about LVR",
        "What plagues current AMM designs?",
        "How do L2 sequencers work?",
        "Do an exhaustive breakdown of the MEV supply chain",
        "What is ePBS?",
        "What is SUAVE?",
        "What are intents?",
        "What are the papers that deal with LVR?",
        "What are solutions to mitigate front-running and sandwich attacks?",
    ]
    for query_str in queries:
        query_input = SYSTEM_MESSAGE.format(question=query_str)
        # TODO 2023-09-29: unsure if the system_message is passed as kwarg frankly.
        #   Even if we format the input query, the ReAct system only logs the question as 'action input'
        #   and seemingly even if the SYSTEM_MESSAGE is very small, the agent still responds to unrelated questions,
        #   even if the sysmessage is a standalone chat message (to which ReAct agent acknowledges).
        # response = chat_engine.chat(SYSTEM_MESSAGE)

        # TODO 2023-09-30: fix agent when it tries to reach search_engine_tool
        response = chat_engine.chat(query_str)
        log_and_store(store_response_partial, query_str, response)
        chat_engine.reset()

    logging.info("Test completed.")
    pass


def run():
    start_logging()
    recreate_index = False
    # embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME_OSS')
    embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME_OPENAI')
    embedding_model_chunk_size = config.EMBEDDING_DIMENSIONS[embedding_model_name]
    chunk_overlap = get_chunk_overlap(embedding_model_chunk_size)
    embedding_model = get_embedding_model(embedding_model_name=embedding_model_name)
    if recreate_index:
        logging.info("RECREATING INDEX")
        # 1. Data loading
        # pdf_links, save_dir = fetch_pdf_list(num_papers=None)
        # download_pdfs(pdf_links, save_dir)
        documents = load_pdfs(directory_path=Path(pdfs_dir))

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
        persist_index(vector_store, index, embedding_model_name, embedding_model_chunk_size, chunk_overlap)
    else:
        logging.info("LOADING INDEX FROM DISK")
        index = load_index_from_disk()

    # TODO 2023-09-29: when iterating on retrieval and LLM, retrieve index from disk instead of re-creating it

    # 7. Retrieve and Query from the Vector Store
    # Now that our ingestion is complete, we can retrieve/query this vector store.
    # NOTE: We can use our high-level VectorStoreIndex abstraction here. See the next section to see how to define retrieval at a lower-level!
    retrieve_and_query_from_vector_store(embedding_model_name=embedding_model_name,
                                         llm_model_name=os.environ.get('LLM_MODEL_NAME_OPENAI'),
                                         chunksize=embedding_model_chunk_size,
                                         chunkoverlap=chunk_overlap,
                                         index=index)

    return index
    pass
    # delete the index to save resources once we are done
    # vector_store.delete(deleteAll=True)


if __name__ == "__main__":
    run()
