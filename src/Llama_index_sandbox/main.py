# https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
# Credits to https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
import logging
import os
from llama_index import ServiceContext

from src.Llama_index_sandbox.constants import INPUT_QUERIES, TEXT_SPLITTER_CHUNK_SIZE, TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE, NUMBER_OF_CHUNKS_TO_RETRIEVE
from src.Llama_index_sandbox.gcs_utils import set_secrets_from_cloud
from src.Llama_index_sandbox.retrieve import get_engine_from_vector_store, ask_questions, get_inference_llm
from src.Llama_index_sandbox.utils import start_logging, get_last_index_embedding_params
import src.Llama_index_sandbox.embed as embed
from src.Llama_index_sandbox.index import load_index_from_disk, create_index


def initialise_chatbot(engine, query_engine_as_tool):
    recreate_index = True
    add_new_transcripts = False
    stream = True
    num_files = None
    similarity_top_k = NUMBER_OF_CHUNKS_TO_RETRIEVE
    text_splitter_chunk_size = TEXT_SPLITTER_CHUNK_SIZE
    text_splitter_chunk_overlap_percentage = TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE

    embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME_OSS')
    # embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME_OPENAI')
    embedding_model = embed.get_embedding_model(embedding_model_name=embedding_model_name)

    llm_model_name = os.environ.get('LLM_MODEL_NAME_OPENAI')
    # llm_model_name = os.environ.get('LLM_MODEL_NAME_OSS')
    llm = get_inference_llm(llm_model_name=llm_model_name)
    service_context: ServiceContext = ServiceContext.from_defaults(llm=llm, embed_model=embedding_model)

    start_logging(f"create_index_{embedding_model_name.split('/')[-1]}_{llm_model_name}_{text_splitter_chunk_size}_{text_splitter_chunk_overlap_percentage}_{similarity_top_k}")
    index_embedding_model_name, index_text_splitter_chunk_size, index_chunk_overlap, vector_space_distance_metric = get_last_index_embedding_params()
    if (not recreate_index) and ((index_embedding_model_name != embedding_model_name.split('/')[-1]) or (index_text_splitter_chunk_size != TEXT_SPLITTER_CHUNK_SIZE) or (index_chunk_overlap != TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE)):
        logging.error(f"The new embedding model parameters are the same as the last ones and we are not recreating the index. Do you want to recreate the index or to revert parameters back?")
        assert False

    if recreate_index:
        index = create_index(embedding_model_name=embedding_model_name,
                             text_splitter_chunk_size=text_splitter_chunk_size,
                             text_splitter_chunk_overlap_percentage=text_splitter_chunk_overlap_percentage,
                             embedding_model=embedding_model,
                             vector_space_distance_metric=vector_space_distance_metric,
                             add_new_transcripts=add_new_transcripts,
                             num_files=num_files)
    else:
        index = load_index_from_disk(service_context)

    # 7. Retrieve and Query from the Vector Store
    # Now that our ingestion is complete, we can retrieve/query this vector store.
    retrieval_engine, query_engine, store_response_partial = get_engine_from_vector_store(embedding_model_name=embedding_model_name,
                                                                                          embedding_model=embedding_model,
                                                                                          llm_model_name=llm_model_name,
                                                                                          service_context=service_context,
                                                                                          TEXT_SPLITTER_CHUNK_SIZE=TEXT_SPLITTER_CHUNK_SIZE,
                                                                                          TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE=TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE,
                                                                                          similarity_top_k=similarity_top_k,
                                                                                          index=index,
                                                                                          engine=engine,
                                                                                          stream=stream,
                                                                                          query_engine_as_tool=query_engine_as_tool)
    return retrieval_engine, query_engine, store_response_partial


def run():
    set_secrets_from_cloud()
    engine = 'chat'
    query_engine_as_tool = True
    reset_chat = True

    logging.info(f"Run parameters: engine={engine}, query_engine_as_tool={query_engine_as_tool}, reset_chat={reset_chat}")

    retrieval_engine, query_engine, store_response_partial = initialise_chatbot(engine=engine, query_engine_as_tool=query_engine_as_tool)
    ask_questions(input_queries=INPUT_QUERIES, retrieval_engine=retrieval_engine, query_engine=query_engine,
                  store_response_partial=store_response_partial, engine=engine, query_engine_as_tool=query_engine_as_tool, reset_chat=reset_chat)
    return retrieval_engine


if __name__ == "__main__":
    run()
