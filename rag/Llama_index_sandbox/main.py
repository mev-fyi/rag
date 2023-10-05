# https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
# Credits to https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html

import os
from pathlib import Path
import logging

import rag.Llama_index_sandbox.config as config
from rag.Llama_index_sandbox.data_ingestion_youtube.load.load import load_video_transcripts
from rag.Llama_index_sandbox.utils import start_logging, timeit
from rag.Llama_index_sandbox import pdfs_dir, video_transcripts_dir
import rag.Llama_index_sandbox.data_ingestion_pdf.load as load_pdf
import rag.Llama_index_sandbox.data_ingestion_pdf.chunk as chunk_pdf
import rag.Llama_index_sandbox.data_ingestion_youtube.chunk as chunk_youtube
import rag.Llama_index_sandbox.embed as embed
from rag.Llama_index_sandbox.index import load_nodes_into_vector_store_create_index, load_index_from_disk, persist_index
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from rag.Llama_index_sandbox.store_response import store_response
from functools import partial
from rag.Llama_index_sandbox.constants import SYSTEM_MESSAGE
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.indices.query.base import BaseQueryEngine


def format_metadata(response):
    title_to_metadata = {}

    for key, meta_info in response.metadata.items():
        title = meta_info.get('title', 'N/A')

        if 'authors' in meta_info:
            authors_list = meta_info.get('authors', 'N/A').split(', ')
            formatted_authors = authors_list[0] + (' et al.' if len(authors_list) > 3 else ', '.join(authors_list[1:]))
        else:
            formatted_authors = None

        if title not in title_to_metadata:
            title_to_metadata[title] = {
                'formatted_authors': formatted_authors,
                'pdf_link': meta_info.get('pdf_link', 'N/A'),
                'release_date': meta_info.get('release_date', 'N/A'),
                'channel_name': meta_info.get('channel_name', 'N/A'),
                'video_link': meta_info.get('video_link', 'N/A'),
                'published_date': meta_info.get('published_date', 'N/A'),
                'chunks_count': 0
            }

        title_to_metadata[title]['chunks_count'] += 1

    # Sorting metadata based on dates (from most recent to oldest)
    sorted_metadata = sorted(title_to_metadata.items(), key=lambda x: (x[1]['release_date'] if x[1]['release_date'] != 'N/A' else x[1]['published_date']), reverse=True)

    formatted_metadata_list = []
    for title, meta in sorted_metadata:
        if meta['formatted_authors']:
            formatted_metadata = f"[Title]: {title}, [Authors]: {meta['formatted_authors']}, [Link]: {meta['pdf_link']}, [Release date]: {meta['release_date']}, [# chunks retrieved]: {meta['chunks_count']}"
        else:
            formatted_metadata = f"[Title]: {title}, [Channel name]: {meta['channel_name']}, [Video Link]: {meta['video_link']}, [Published date]: {meta['published_date']}, [# chunks retrieved]: {meta['chunks_count']}"

        formatted_metadata_list.append(formatted_metadata)

    # Joining all formatted metadata strings with a newline
    all_formatted_metadata = '\n'.join(formatted_metadata_list)
    return all_formatted_metadata



def log_and_store(store_response_fn, query_str, response):
    all_formatted_metadata = format_metadata(response)
    logging.info(f"[Shown to client] The answer to [{query_str}] is: \n\n```\n{response}\n```\n\nWith sources: \n{all_formatted_metadata}")
    # store_response_fn(query_str, response)


def get_chat_engine(index, service_context, chat_mode="react", verbose=True, similarity_top_k=5):
    # TODO 2023-09-29: make sure the temperature is set to zero for consistent results
    # TODO 2023-09-29: creating a (react) chat engine from an index transforms that
    #  query as a tool and passes it to the agent under the hood. That query tool can receive a description.
    #  We need to determine (1) if we pass several query engines as tool or build a massive single one (cost TBD),
    #  and (2) if we pass a description to the query tool and what is the expected retrieval impact from having a description versus not.

    # TODO 2023-09-29: add system prompt to agent. BUT it is an input to OpenAI agent but not React Agent!
    #   OpenAI agent has prefix_messages in its constructor, but React Agent does not.
    chat_engine = index.as_chat_engine(chat_mode=chat_mode,
                                       verbose=verbose,
                                       similarity_top_k=similarity_top_k,
                                       service_context=service_context,
                                       system_prompt=SYSTEM_MESSAGE)
    return chat_engine


def get_query_engine(index, service_context, verbose=True, similarity_top_k=5):
    return index.as_query_engine(similarity_top_k=similarity_top_k, service_context=service_context, verbose=verbose)


@timeit
def retrieve_and_query_from_vector_store(embedding_model_name, llm_model_name, chunksize, chunkoverlap, index, engine='chat', similarity_top_k=10):
    # TODO 2023-09-29: we need to set in stone an accurate baseline evaluation using ReAct agent.
    #   To achieve this we need to save intermediary Response objects to make sure we can distill results and have access to nodes and chunks used for the reasoning
    # TODO 2023-09-29: determine how we should structure our indexes per document type
    service_context = ServiceContext.from_defaults(llm=OpenAI(model=llm_model_name))
    # create partial store_response with everything but the query_str and response
    store_response_partial = partial(store_response, embedding_model_name, llm_model_name, chunksize, chunkoverlap)

    # chat_engine = index.as_chat_engine(chat_mode="react", verbose=True, similarity_top_k=5, service_context=service_context)
    if engine == 'chat':
        # create an LLM response based on chain of thoughts to return the final result
        # TODO 2023-10-05: tune timeout and max_tokens
        retrieval_engine = get_chat_engine(index, service_context, chat_mode="react", verbose=True, similarity_top_k=similarity_top_k)
    elif engine == 'query':
        retrieval_engine = get_query_engine(index=index, service_context=service_context, verbose=True, similarity_top_k=similarity_top_k)
    else:
        assert False, f"Please specify a retrieval engine amongst ['chat', 'query'], current input: {engine}"

    queries = [
        # "What is red teaming in AI",  # Should refuse to respond,
        "Tell me about LVR",
        "What plagues current AMM designs?",
        "How do L2 sequencers work?",
        "Do an exhaustive breakdown of the MEV supply chain",
        "What is ePBS?",
        "What is SUAVE?",
        "What are intents?",
        "What are the papers that deal with LVR?",
        "What are solutions to mitigate front-running and sandwich attacks?",
        "What content discusses L2 sequencers?",
        "What content discusses L two s sequencers?",
        "What content discusses SUAVE?",
        "Tell me about transaction ordering on L two s",
        "Can you tell me how the definition of MEV evolved over the years?",
        "What are videos that discuss OFAs?",
        "Cite all the sources you have about Tim Roughgarden"
    ]
    for query_str in queries:
        query_input = SYSTEM_MESSAGE.format(question=query_str)
        # TODO 2023-09-29: unsure if the system_message is passed as kwarg frankly.
        #   Even if we format the input query, the ReAct system only logs the question as 'action input'
        #   and seemingly even if the SYSTEM_MESSAGE is very small, the agent still responds to unrelated questions,
        #   even if the sysmessage is a standalone chat message (to which ReAct agent acknowledges).
        # response = chat_engine.chat(SYSTEM_MESSAGE)
        # TODO 2023-10-05: understand why metadata of videos is not written at all
        if isinstance(retrieval_engine, BaseChatEngine):
            response = retrieval_engine.chat(query_str)
            # retrieval_engine.reset()
        elif isinstance(retrieval_engine, BaseQueryEngine):
            # TODO 2023-10-05: fix the retrieval of the index. Currently if we load it back i.e. use the ID to fetch the index, it does not work?
            #  and only re-running the whole pipeline works. perhaps a question of upgrading to paying. TBD. if cost is $0.7 an hour versus ~$0.50 for each embed run (OpenAI).
            logging.info(f"\nQuerying index with query:    [{query_str}]")
            response = retrieval_engine.query(query_str)
        else:
            logging.error(f"Please specify a retrieval engine amongst ['chat', 'query'], current input: {engine}")
            assert False

        log_and_store(store_response_partial, query_str, response)
        # TODO 2023-10-05 [RETRIEVAL]: in particular for chunks from youtube videos, we might want to expand the window from which it retrieved the chunk
        # TODO 2023-10-05 [RETRIEVAL]: since many chunks can be retrieved from a single youtube video, what should be the returned timestamp to these references? should we return them all?
        # TODO 2023-10-05 [RETRIEVAL]: add weights such that responses from older sources have less importance in the answer
        # TODO 2023-10-05 [RETRIEVAL]: should we weight more a person which is an author and has a paper?


        # TODO 2023-10-05: send the resulting chain of though to gpt3.5 turbo
        # TODO 2023-10-05: update the chain of thought to display each file and chunk used for the reasoning
        # TODO 2023-10-05: return the metadata of each file and chunk used for the reasoning for referencing

    logging.info("Test completed.")
    return retrieval_engine
    # pass


def run():
    start_logging()
    recreate_index = False
    # embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME_OSS')
    embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME_OPENAI')
    embedding_model_chunk_size = config.EMBEDDING_DIMENSIONS[embedding_model_name]
    chunk_overlap = chunk_pdf.get_chunk_overlap(embedding_model_chunk_size)
    embedding_model = embed.get_embedding_model(embedding_model_name=embedding_model_name)
    if recreate_index:
        logging.info("RECREATING INDEX")
        # 1. Data loading
        # pdf_links, save_dir = fetch_pdf_list(num_papers=None)
        # download_pdfs(pdf_links, save_dir)
        documents_pdfs = load_pdf.load_pdfs(directory_path=Path(pdfs_dir))  # [:1]
        documents_youtube = load_video_transcripts(directory_path=Path(video_transcripts_dir))  # [:5]

        # 2. Data chunking / text splitter
        text_chunks_pdfs, doc_idxs_pdfs = chunk_pdf.chunk_documents(documents_pdfs, chunk_size=embedding_model_chunk_size)
        text_chunks_youtube, doc_idxs_youtube = chunk_youtube.chunk_documents(documents_youtube, chunk_size=embedding_model_chunk_size)

        # 3. Manually Construct Nodes from Text Chunks
        nodes_pdf = embed.construct_node(text_chunks_pdfs, documents_pdfs, doc_idxs_pdfs)
        nodes_youtube = embed.construct_node(text_chunks_youtube, documents_youtube, doc_idxs_youtube)

        # [Optional] 4. Extract Metadata from each Node by performing LLM calls to fetch Title.
        #        We extract metadata from each Node using our Metadata extractors.
        #        This will add more metadata to each Node.
        # nodes = enrich_nodes_with_metadata_via_llm(nodes)

        # 5. Generate Embeddings for each Node
        embed.generate_embeddings(nodes_pdf, embedding_model)
        embed.generate_embeddings(nodes_youtube, embedding_model)
        nodes = nodes_pdf + nodes_youtube

        # 6. Load Nodes into a Vector Store
        # We now insert these nodes into our PineconeVectorStore.
        # NOTE: We skip the VectorStoreIndex abstraction, which is a higher-level abstraction
        # that handles ingestion as well. We use VectorStoreIndex in the next section to fast-trak retrieval/querying.
        index = load_nodes_into_vector_store_create_index(nodes, embedding_model_chunk_size)
        persist_index(index, embedding_model_name, embedding_model_chunk_size, chunk_overlap)
    else:
        index = load_index_from_disk()

    # TODO 2023-09-29: when iterating on retrieval and LLM, retrieve index from disk instead of re-creating it

    # 7. Retrieve and Query from the Vector Store
    # Now that our ingestion is complete, we can retrieve/query this vector store.
    # NOTE: We can use our high-level VectorStoreIndex abstraction here. See the next section to see how to define retrieval at a lower-level!
    retrieval_engine = retrieve_and_query_from_vector_store(embedding_model_name=embedding_model_name,
                                                            llm_model_name=os.environ.get('LLM_MODEL_NAME_OPENAI'),
                                                            chunksize=embedding_model_chunk_size,
                                                            chunkoverlap=chunk_overlap,
                                                            index=index,
                                                            engine='query')

    return retrieval_engine
    pass
    # delete the index to save resources once we are done
    # vector_store.delete(deleteAll=True)


if __name__ == "__main__":
    run()
