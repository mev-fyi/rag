# https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
# Credits to https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
import logging
import os

import src.Llama_index_sandbox.config as config
from src.Llama_index_sandbox.constants import INPUT_QUERIES, CHUNK_SIZE_PERCENTAGE, CHUNK_OVERLAP_PERCENTAGE
from src.Llama_index_sandbox.retrieve import get_engine_from_vector_store, ask_questions
from src.Llama_index_sandbox.utils import start_logging, get_last_index_embedding_params
import src.Llama_index_sandbox.data_ingestion_pdf.chunk as chunk_pdf
import src.Llama_index_sandbox.embed as embed
from src.Llama_index_sandbox.index import load_index_from_disk, create_index


def initialise_chatbot(engine, query_engine_as_tool):
    start_logging()

    recreate_index = True
    stream = True

    # embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME_OSS')
    embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME_OPENAI')
    embedding_model_chunk_size = int(config.EMBEDDING_DIMENSIONS[embedding_model_name] * CHUNK_SIZE_PERCENTAGE/100)
    chunk_overlap = chunk_pdf.get_chunk_overlap(CHUNK_OVERLAP_PERCENTAGE, embedding_model_chunk_size)
    embedding_model = embed.get_embedding_model(embedding_model_name=embedding_model_name)

    index_embedding_model_name, index_last_embedding_model_chunk_size, index_chunk_overlap = get_last_index_embedding_params()
    if (not recreate_index) and ((index_embedding_model_name == embedding_model_name) or (index_last_embedding_model_chunk_size == embedding_model_chunk_size) or (index_chunk_overlap == chunk_overlap)):
        logging.error(f"The new embedding model parameters are the same as the last ones and we are not recreating the index. Do you want to recreate the index or to revert parameters back?")
        assert False

    if recreate_index:
        index = create_index(embedding_model_name=embedding_model_name,
                             CHUNK_SIZE_PERCENTAGE=CHUNK_SIZE_PERCENTAGE,
                             embedding_model_chunk_size=embedding_model_chunk_size,
                             CHUNK_OVERLAP_PERCENTAGE=CHUNK_OVERLAP_PERCENTAGE,
                             embedding_model=embedding_model)
    else:
        index = load_index_from_disk()

    # 7. Retrieve and Query from the Vector Store
    # Now that our ingestion is complete, we can retrieve/query this vector store.
    retrieval_engine, query_engine, store_response_partial = get_engine_from_vector_store(embedding_model_name=embedding_model_name,
                                                                                          llm_model_name=os.environ.get('LLM_MODEL_NAME_OPENAI'),
                                                                                          chunksize=embedding_model_chunk_size,
                                                                                          chunkoverlap=chunk_overlap,
                                                                                          index=index,
                                                                                          engine=engine,
                                                                                          stream=stream,
                                                                                          query_engine_as_tool=query_engine_as_tool)
    return retrieval_engine, query_engine, store_response_partial


def run():
    engine = 'chat'
    query_engine_as_tool = True

    retrieval_engine, query_engine, store_response_partial = initialise_chatbot(engine=engine, query_engine_as_tool=query_engine_as_tool)
    ask_questions(input_queries=INPUT_QUERIES, retrieval_engine=retrieval_engine, query_engine=query_engine,
                  store_response_partial=store_response_partial, engine=engine, query_engine_as_tool=query_engine_as_tool)
    return retrieval_engine


if __name__ == "__main__":
    run()
