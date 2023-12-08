# https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
# Credits to https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
import logging
import os
from llama_index import ServiceContext

from src.Llama_index_sandbox.constants import INPUT_QUERIES
from src.Llama_index_sandbox.evaluation.config import Config
from src.Llama_index_sandbox.gcs_utils import set_secrets_from_cloud
from src.Llama_index_sandbox.retrieve import get_engine_from_vector_store, ask_questions, get_inference_llm
from src.Llama_index_sandbox.utils import start_logging, get_last_index_embedding_params
import src.Llama_index_sandbox.embed as embed
from src.Llama_index_sandbox.index import load_index_from_disk, create_index


def initialise_chatbot(engine, query_engine_as_tool, recreate_index):
    add_new_transcripts = False
    stream = True
    config = Config()
    num_files = config.num_files
    similarity_top_k = config.NUM_CHUNKS_RETRIEVED[0]
    text_splitter_chunk_size = config.CHUNK_SIZES[0]
    text_splitter_chunk_overlap_percentage = config.CHUNK_OVERLAPS[0]

    embedding_model_name = config.EMBEDDING_MODELS[0]
    # embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME_OPENAI')
    embedding_model = embed.get_embedding_model(embedding_model_name=embedding_model_name)

    llm_model_name = config.INFERENCE_MODELS[0]
    # llm_model_name = os.environ.get('LLM_MODEL_NAME_OSS')
    llm = get_inference_llm(llm_model_name=llm_model_name)
    service_context: ServiceContext = ServiceContext.from_defaults(llm=llm, embed_model=embedding_model)

    start_logging(f"create_index_{embedding_model_name.split('/')[-1]}_{llm_model_name}_{text_splitter_chunk_size}_{text_splitter_chunk_overlap_percentage}_{similarity_top_k}")
    index_embedding_model_name, index_text_splitter_chunk_size, index_chunk_overlap, vector_space_distance_metric = get_last_index_embedding_params()
    logging.info(f"recreate_index: {recreate_index}, index_embedding_model_name={index_embedding_model_name}, index_text_splitter_chunk_size={index_text_splitter_chunk_size}, index_chunk_overlap={index_chunk_overlap}, vector_space_distance_metric={vector_space_distance_metric}")
    logging.info(f"index_embedding_model_name: {index_embedding_model_name}, index_text_splitter_chunk_size: {index_text_splitter_chunk_size}, index_chunk_overlap: {index_chunk_overlap}, vector_space_distance_metric: {vector_space_distance_metric}")
    if (not recreate_index) and ((index_embedding_model_name != embedding_model_name.split('/')[-1]) or (index_text_splitter_chunk_size != text_splitter_chunk_size) or (index_chunk_overlap != text_splitter_chunk_overlap_percentage)):
        logging.error(f"The new embedding model parameters are different from the last ones and we are not recreating the index. Do you want to recreate the index or to revert parameters back?")
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
    log_name = f"{embedding_model_name.split('/')[-1]}_{llm_model_name}_{text_splitter_chunk_size}_{text_splitter_chunk_overlap_percentage}"
    start_logging(f"ask_questions_{log_name}_{similarity_top_k}")
    retrieval_engine, query_engine, store_response_partial = get_engine_from_vector_store(embedding_model_name=embedding_model_name,
                                                                                          embedding_model=embedding_model,
                                                                                          llm_model_name=llm_model_name,
                                                                                          service_context=service_context,
                                                                                          text_splitter_chunk_size=text_splitter_chunk_size,
                                                                                          text_splitter_chunk_overlap_percentage=text_splitter_chunk_overlap_percentage,
                                                                                          similarity_top_k=similarity_top_k,
                                                                                          index=index,
                                                                                          engine=engine,
                                                                                          stream=stream,
                                                                                          query_engine_as_tool=query_engine_as_tool,
                                                                                          log_name=log_name)
    return retrieval_engine, query_engine, store_response_partial, config


def run():
    if not os.environ.get('ENVIRONMENT') == 'LOCAL':
        set_secrets_from_cloud()
    engine = 'chat'
    query_engine_as_tool = True
    recreate_index = False
    chat_history = []

    logging.info(f"Run parameters: engine={engine}, query_engine_as_tool={query_engine_as_tool}")

    retrieval_engine, query_engine, store_response_partial, config = initialise_chatbot(engine=engine, query_engine_as_tool=query_engine_as_tool, recreate_index=recreate_index)
    ask_questions(input_queries=INPUT_QUERIES[:2], retrieval_engine=retrieval_engine, query_engine=query_engine,
                  store_response_partial=store_response_partial, engine=engine, query_engine_as_tool=query_engine_as_tool, chat_history=chat_history, reset_chat=config.reset_chat)
    return retrieval_engine


if __name__ == "__main__":
    run()
