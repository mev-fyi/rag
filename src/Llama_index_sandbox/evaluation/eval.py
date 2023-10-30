# https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
# Credits to https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
import logging
import os
from llama_index import ServiceContext
from itertools import product

from src.Llama_index_sandbox.constants import TEXT_SPLITTER_CHUNK_SIZE, TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE, NUMBER_OF_CHUNKS_TO_RETRIEVE, EVALUATION_INPUT_QUERIES
from src.Llama_index_sandbox.evaluation.evaluation_constants import CHUNK_SIZES, CHUNK_OVERLAPS, NUM_CHUNKS_RETRIEVED, EMBEDDING_MODELS, INFERENCE_MODELS
from src.Llama_index_sandbox.retrieve import get_engine_from_vector_store, ask_questions, get_inference_llm
import src.Llama_index_sandbox.embed as embed
from src.Llama_index_sandbox.index import load_index_from_disk, create_index


def initialise_chatbot(engine, query_engine_as_tool, params):
    recreate_index = params["recreate_index"]
    similarity_top_k = params["similarity_top_k"]
    text_splitter_chunk_size = params["text_splitter_chunk_size"]
    text_splitter_chunk_overlap_percentage = params["text_splitter_chunk_overlap_percentage"]
    embedding_model_name = params["embedding_model_name"]
    embedding_model = params["embedding_model"]
    llm_model_name = params["llm_model_name"]
    add_new_transcripts = params["add_new_transcripts"]
    num_files = params["num_files"]
    stream = params["stream"]

    # Create the service context inside this function based on the current combination
    llm = get_inference_llm(llm_model_name=llm_model_name)
    service_context: ServiceContext = ServiceContext.from_defaults(llm=llm, embed_model=embedding_model)

    if recreate_index:
        index = create_index(embedding_model_name=embedding_model_name,
                             text_splitter_chunk_size=text_splitter_chunk_size,
                             text_splitter_chunk_overlap_percentage=text_splitter_chunk_overlap_percentage,
                             embedding_model=embedding_model,
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
    engine = 'chat'
    query_engine_as_tool = True
    reset_chat = True
    add_new_transcripts = False
    stream = True
    num_files = None

    # Lists
    similarity_top_k_list = NUM_CHUNKS_RETRIEVED
    text_splitter_chunk_size_list = CHUNK_SIZES
    text_splitter_chunk_overlap_percentage_list = CHUNK_OVERLAPS
    embedding_model_name_list = EMBEDDING_MODELS
    embedding_model_list = [embed.get_embedding_model(embedding_model_name=e) for e in embedding_model_name_list]
    llm_model_name_list = INFERENCE_MODELS

    # Combinations affecting the index creation
    index_combinations = product(text_splitter_chunk_size_list, text_splitter_chunk_overlap_percentage_list, embedding_model_name_list, embedding_model_list)

    for index_comb in index_combinations:
        # Unpacking
        text_splitter_chunk_size, text_splitter_chunk_overlap_percentage, embedding_model_name, embedding_model = index_comb

        # Check if recreate_index needs to be True (since only index params are looped here)
        recreate_index = text_splitter_chunk_size != TEXT_SPLITTER_CHUNK_SIZE or text_splitter_chunk_overlap_percentage != TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE or embedding_model_name != embedding_model_name_list[0]

        # Now, for the given index_comb, loop through llm and top_k combinations
        inference_combinations = product(llm_model_name_list, similarity_top_k_list)
        for llm_model_name, similarity_top_k in inference_combinations:
            # Combine to get full combination
            full_combination = {
                "recreate_index": recreate_index,
                "similarity_top_k": similarity_top_k,
                "text_splitter_chunk_size": text_splitter_chunk_size,
                "text_splitter_chunk_overlap_percentage": text_splitter_chunk_overlap_percentage,
                "embedding_model_name": embedding_model_name,
                "embedding_model": embedding_model,
                "llm_model_name": llm_model_name,
                "add_new_transcripts": add_new_transcripts,
                "num_files": num_files,
                "stream": stream
            }

            retrieval_engine, query_engine, store_response_partial = initialise_chatbot(engine=engine, query_engine_as_tool=query_engine_as_tool, params=full_combination)

            ask_questions(input_queries=EVALUATION_INPUT_QUERIES, retrieval_engine=retrieval_engine, query_engine=query_engine,
                          store_response_partial=store_response_partial, engine=engine, query_engine_as_tool=query_engine_as_tool, reset_chat=reset_chat)


if __name__ == "__main__":
    run()
