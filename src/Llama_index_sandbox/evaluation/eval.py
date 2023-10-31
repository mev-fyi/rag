# https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
# Credits to https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
from llama_index import ServiceContext
from itertools import product
from typing import Tuple, Dict, Any

from src.Llama_index_sandbox.constants import TEXT_SPLITTER_CHUNK_SIZE, TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE, EVALUATION_INPUT_QUERIES
from src.Llama_index_sandbox.evaluation.config import Config
from src.Llama_index_sandbox.retrieve import get_engine_from_vector_store, ask_questions, get_inference_llm
from src.Llama_index_sandbox.index import load_index_from_disk, create_index
from src.Llama_index_sandbox.utils import start_logging
from src.Llama_index_sandbox import globals as glb


def get_or_create_index(params: Dict[str, Any]) -> Tuple[Any, ServiceContext]:
    """
    Get or create an index based on provided parameters.

    Args:
    - params (dict): Dictionary of parameters including embedding model, chunk size, etc.

    Returns:
    - Tuple containing the index and the service context.
    """
    embedding_model = params["embedding_model"]
    llm_model_name = params["llm_model_name"]

    # Create the service context inside this function based on the current combination
    llm = get_inference_llm(llm_model_name=llm_model_name)
    service_context: ServiceContext = ServiceContext.from_defaults(llm=llm, embed_model=embedding_model)
    recreate_index = params["recreate_index"]
    if recreate_index:
        index = create_index(embedding_model_name=params["embedding_model_name"],
                             text_splitter_chunk_size=params["text_splitter_chunk_size"],
                             text_splitter_chunk_overlap_percentage=params["text_splitter_chunk_overlap_percentage"],
                             embedding_model=params["embedding_model"],
                             add_new_transcripts=params["add_new_transcripts"],
                             num_files=params["num_files"])
    else:
        index = load_index_from_disk(service_context)
    return index, service_context


def initialise_chatbot(engine, query_engine_as_tool, index, service_context, params):
    similarity_top_k = params["similarity_top_k"]
    embedding_model_name = params["embedding_model_name"]
    embedding_model = params["embedding_model"]
    llm_model_name = params["llm_model_name"]
    stream = params["stream"]

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


def run(config: Config):
    for index_comb in config.get_full_combinations():
        text_splitter_chunk_size, text_splitter_chunk_overlap_percentage, embedding_model_name, embedding_model, llm_model_name = index_comb
        index_params = config.get_index_params(text_splitter_chunk_size, text_splitter_chunk_overlap_percentage, embedding_model_name, embedding_model, llm_model_name)

        start_logging(f"create_index_{embedding_model_name.split('/')[-1]}_{llm_model_name}_{text_splitter_chunk_size}_{text_splitter_chunk_overlap_percentage}")
        index, service_context = get_or_create_index(index_params)
        for llm_model_name, similarity_top_k in product(config.INFERENCE_MODELS, config.NUM_CHUNKS_RETRIEVED):
            inference_params = config.get_inference_params(llm_model_name, similarity_top_k, text_splitter_chunk_size, text_splitter_chunk_overlap_percentage, embedding_model_name, embedding_model)
            retrieval_engine, query_engine, store_response_partial = initialise_chatbot(engine=config.engine, query_engine_as_tool=config.query_engine_as_tool, index=index, service_context=service_context, params=inference_params)

            # write NUMBER_OF_CHUNKS_TO_RETRIEVE as global scope
            glb.NUMBER_OF_CHUNKS_TO_RETRIEVE = similarity_top_k

            start_logging(f"ask_questions_{embedding_model_name.split('/')[-1]}_{llm_model_name}_{text_splitter_chunk_size}_{text_splitter_chunk_overlap_percentage}_{similarity_top_k}")
            ask_questions(input_queries=EVALUATION_INPUT_QUERIES, retrieval_engine=retrieval_engine, query_engine=query_engine, store_response_partial=store_response_partial, engine=config.engine, query_engine_as_tool=config.query_engine_as_tool, reset_chat=config.reset_chat)


if __name__ == "__main__":
    config = Config()
    run(config)
