# https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
# Credits to https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
import os

import rag.Llama_index_sandbox.config as config
from rag.Llama_index_sandbox.constants import INPUT_QUERIES
from rag.Llama_index_sandbox.retrieve import get_engine_from_vector_store, ask_questions
from rag.Llama_index_sandbox.utils import start_logging
import rag.Llama_index_sandbox.data_ingestion_pdf.chunk as chunk_pdf
import rag.Llama_index_sandbox.embed as embed
from rag.Llama_index_sandbox.index import load_index_from_disk, create_index


def initialise_chatbot(engine):
    start_logging()
    recreate_index = False
    # embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME_OSS')
    embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME_OPENAI')
    embedding_model_chunk_size = config.EMBEDDING_DIMENSIONS[embedding_model_name]
    chunk_overlap = chunk_pdf.get_chunk_overlap(embedding_model_chunk_size)
    embedding_model = embed.get_embedding_model(embedding_model_name=embedding_model_name)
    if recreate_index:
        index = create_index(embedding_model_name=embedding_model_name,
                             embedding_model_chunk_size=embedding_model_chunk_size,
                             chunk_overlap=chunk_overlap,
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
                                                                                          engine=engine)
    return retrieval_engine, query_engine, store_response_partial


def run():
    engine = 'chat'
    retrieval_engine, query_engine, store_response_partial = initialise_chatbot(engine=engine)
    # TODO 2023-10-13: if we solely ask a question about a previous answer, but that we pass this question to the query engine,
    #   then any successive result would not be relevant. perhaps we need a classifier to determine whether to pass the question to the query engine or not.
    ask_questions(input_queries=INPUT_QUERIES, retrieval_engine=retrieval_engine, query_engine=query_engine,
                  store_response_partial=store_response_partial, engine=engine)
    # If paid: delete the index to save resources once we are done ($0.70 per hr versus ~$0.50 to create it)
    # vector_store.delete(deleteAll=True)
    return retrieval_engine


if __name__ == "__main__":
    run()
