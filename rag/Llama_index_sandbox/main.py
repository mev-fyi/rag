# Credits to https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html

import os
from pathlib import Path
import logging
import rag.config as config
from rag.Llama_index_sandbox.utils import start_logging, timeit
from rag.Llama_index_sandbox import pdfs_dir
from rag.Llama_index_sandbox.data_ingestion_pdf.load import load_pdfs, fetch_pdf_list, download_pdfs
from rag.Llama_index_sandbox.data_ingestion_pdf.chunk import chunk_documents
from rag.Llama_index_sandbox.data_ingestion_pdf.embed import generate_embeddings, get_embedding_model, construct_node
from rag.Llama_index_sandbox.data_ingestion_pdf.index import initialise_vector_store, load_nodes_into_vector_store_create_index
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from rag.Llama_index_sandbox.store_response import store_response
from functools import partial


def log_and_store(store_response_fn, query_str, response):
    logging.info(response)
    store_response_fn(query_str, response)


@timeit
def retrieve_and_query_from_vector_store(embedding_model_name, llm_model_name, chunksize, chunkoverlap, index):
    service_context = ServiceContext.from_defaults(llm=OpenAI(model=llm_model_name))
    # create partial store_response with everything but the query_str and response
    store_response_partial = partial(store_response, embedding_model_name, llm_model_name, chunksize, chunkoverlap)

    query_engine = index.as_query_engine(similarity_top_k=5, service_context=service_context)
    # TODO 2023-09-27: if the question is totally unrelated, should the response be empty?
    query_str = "Can you tell me about the key concepts for safety finetuning"
    response = query_engine.query(query_str)
    log_and_store(store_response_partial, query_str, response)

    query_str = "Tell me about LVR"
    response = query_engine.query(query_str)
    log_and_store(store_response_partial, query_str, response)

    query_str = "What plagues current AMM designs?"
    response = query_engine.query(query_str)
    log_and_store(store_response_partial, query_str, response)

    # TODO 2023-09-27: improve the response engine with react agent chatbot.

    # chat_engine = index.as_chat_engine(chat_mode=ChatMode.REACT, verbose=True)
    # response = chat_engine.chat("Hi")
    pass


def run():
    start_logging()
    # 1. Data loading
    # pdf_links, save_dir = fetch_pdf_list(num_papers=None)
    # download_pdfs(pdf_links, save_dir)
    documents = load_pdfs(directory_path=Path(pdfs_dir))

    # embedding_model_str = os.environ.get('EMBEDDING_MODEL_NAME_OSS')
    embedding_model_str = os.environ.get('EMBEDDING_MODEL_NAME_OPENAI')
    embedding_model_chunk_size = config.EMBEDDING_DIMENSIONS[embedding_model_str]
    embedding_model = get_embedding_model(embedding_model_name=embedding_model_str)

    # 2. Data chunking / text splitter
    text_chunks, doc_idxs, chunk_overlap = chunk_documents(documents, chunk_size=embedding_model_chunk_size)

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
    retrieve_and_query_from_vector_store(embedding_model_name=embedding_model_str,
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
