import logging
from functools import partial
from typing import Optional, Type

from llama_index import VectorStoreIndex, ServiceContext
from llama_index.agent import ReActAgent
from llama_index.agent.react.formatter import ReActChatFormatter
from llama_index.agent.react.output_parser import ReActOutputParser

from llama_index.callbacks import CallbackManager
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llms import OpenAI
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.tools import QueryEngineTool

from rag.Llama_index_sandbox.constants import OPENAI_MODEL_NAME, LLM_TEMPERATURE, SYSTEM_MESSAGE, INPUT_QUERIES, QUERY_TOOL_RESPONSE
from rag.Llama_index_sandbox.store_response import store_response
from rag.Llama_index_sandbox.utils import timeit


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
    msg = f"The answer to [{query_str}] is: \n\n```\n{response}\n\n\nFetched based on the following sources/content: \n{all_formatted_metadata}\n```"
    logging.info(f"[Shown to client] {msg}")
    return msg
    # store_response_fn(query_str, response)


def get_query_engine(index, service_context, verbose=True, similarity_top_k=5):
    return index.as_query_engine(similarity_top_k=similarity_top_k, service_context=service_context, verbose=verbose)


def get_inference_llm(temperature,
                      callback_manager: Optional[CallbackManager] = None,
                      max_tokens: Optional[int] = None,
                      llm: Optional[OpenAI] = None,
                      ):
    if callback_manager is not None:
        llm.callback_manager = callback_manager
    return llm or OpenAI(model=OPENAI_MODEL_NAME, temperature=temperature, max_tokens=max_tokens, callback_manager=callback_manager)


def get_chat_engine(index: VectorStoreIndex,
                    service_context: ServiceContext,
                    chat_mode: str = "react",
                    verbose: bool = True,
                    similarity_top_k: int = 5,
                    max_iterations: int = 10,
                    memory: Optional[BaseMemory] = None,
                    memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
                    temperature=LLM_TEMPERATURE):
    # NOTE 2023-09-29: creating a (react) chat engine from an index transforms that
    #  query as a tool and passes it to the agent under the hood. That query tool can receive a description.
    #  We need to determine (1) if we pass several query engines as tool or build a massive single one (cost TBD),
    #  and (2) if we pass a description to the query tool and what is the expected retrieval impact from having a description versus not.

    # NOTE: 2023-09-29: add system prompt to agent. BUT it is an input to OpenAI agent but not React Agent!
    #   OpenAI agent has prefix_messages in its constructor, but React Agent does not. Is adding System prompt to chat history good enough?

    query_engine = get_query_engine(index=index, service_context=service_context, verbose=verbose, similarity_top_k=similarity_top_k)
    query_engine_tool = QueryEngineTool.from_defaults(query_engine=query_engine)
    react_chat_formatter: Optional[ReActChatFormatter] = None  # NOTE 2023-10-06: to configure
    output_parser: Optional[ReActOutputParser] = None  # NOTE 2023-10-06: to configure
    callback_manager: Optional[CallbackManager] = None  # NOTE 2023-10-06: to configure
    # chat_history = [SYSTEM_MESSAGE]  # TODO 2023-10-06: to configure and make sure its the good practice
    chat_history = []  # TODO 2023-10-06: to configure and make sure its the good practice

    llm = service_context.llm
    max_tokens: Optional[int] = None  # NOTE 2023-10-05: tune timeout and max_tokens
    llm = get_inference_llm(temperature=temperature, callback_manager=callback_manager, max_tokens=max_tokens, llm=llm)
    memory = memory or memory_cls.from_defaults(chat_history=chat_history, llm=llm)

    # TODO 2023-09-29: we need to set in stone an accurate baseline evaluation using ReAct agent.
    #   To achieve this we need to save intermediary Response objects to make sure we can distill
    #   results and have access to nodes and chunks used for the reasoning

    return ReActAgent.from_tools(
        # tools=[query_engine_tool],
        tools=[],
        llm=llm,
        max_iterations=max_iterations,
        memory=memory,
        verbose=verbose,
    )


@timeit
def retrieve_and_query_from_vector_store(embedding_model_name: str,
                                         llm_model_name: str,
                                         chunksize: int,
                                         chunkoverlap: int,
                                         index: VectorStoreIndex,
                                         engine='chat',
                                         similarity_top_k=10):

    # TODO 2023-09-29: determine how we should structure our indexes per document type
    service_context: ServiceContext = ServiceContext.from_defaults(llm=OpenAI(model=llm_model_name))
    # create partial store_response with everything but the query_str and response
    store_response_partial = partial(store_response, embedding_model_name, llm_model_name, chunksize, chunkoverlap)

    if engine == 'chat':
        retrieval_engine = get_chat_engine(index=index, service_context=service_context, chat_mode="react", verbose=True, similarity_top_k=similarity_top_k)
        retrieval_engine.chat(SYSTEM_MESSAGE)
        query_engine = get_query_engine(index=index, service_context=service_context, verbose=True, similarity_top_k=similarity_top_k)
    elif engine == 'query':
        retrieval_engine = get_query_engine(index=index, service_context=service_context, verbose=True, similarity_top_k=similarity_top_k)
    else:
        assert False, f"Please specify a retrieval engine amongst ['chat', 'query'], current input: {engine}"

    for query_str in INPUT_QUERIES:
        # TODO 2023-10-08: add the metadata filters  # https://docs.pinecone.io/docs/metadata-filtering#querying-an-index-with-metadata-filters
        if isinstance(retrieval_engine, BaseChatEngine):
            # TODO 2023-10-07 [RETRIEVAL]: prioritise fetching chunks and metadata from CoT agent
            response = query_engine.query(query_str)
            str_response = log_and_store(store_response_partial, query_str, response)

            str_response = QUERY_TOOL_RESPONSE.format(question=query_str, response=str_response)
            logging.info(f"Message passed to chat engine:    \n\n[{str_response}]")
            response = retrieval_engine.chat(str_response)
            logging.info("Chatting with response:    [{response}]".format(response=response))
            # retrieval_engine.reset()

        elif isinstance(retrieval_engine, BaseQueryEngine):
            logging.info(f"Querying index with query:    [{query_str}]")
            response = retrieval_engine.query(query_str)
            log_and_store(store_response_partial, query_str, response)
        else:
            logging.error(f"Please specify a retrieval engine amongst ['chat', 'query'], current input: {engine}")
            assert False

        # TODO 2023-10-05 [RETRIEVAL]: in particular for chunks from youtube videos, we might want
        #   to expand the window from which it retrieved the chunk
        # TODO 2023-10-05 [RETRIEVAL]: since many chunks can be retrieved from a single youtube video,
        #   what should be the returned timestamp to these references? should we return them all? return the one with highest score?
        # TODO 2023-10-05 [RETRIEVAL]: add weights such that responses from older sources have less importance in the answer
        # TODO 2023-10-05 [RETRIEVAL]: should we weight more a person which is an author and has a paper?
        # TODO 2023-10-07 [RETRIEVAL]: ADD metadata filtering e.g. "only video" or "only papers", or "from this author", or "from this channel", or "from 2022 and 2023" etc
        # TODO 2023-10-07 [RETRIEVAL]: in the chat format, is the rag system keeping in memory the previous retrieved chunks? e.g. if an answer is too short can it develop it further?
        # TODO 2023-10-07 [RETRIEVAL]: should we allow the external user to tune the top-k retrieved chunks? the temperature?
        # TODO 2023-10-07 [RETRIEVAL]: usually when asked for resources its not that performant and might at best return a single resource.

        # TODO 2023-10-09 [RETRIEVAL]: use metadata tags for users to choose amongst LVR, Intents, MEV, etc such that it can increase the result speed (and likely accuracy)
        #  and this upfront work is likely a low hanging fruit relative to payoff.
        # TODO 2023-10-09 [RETRIEVAL]: try non-ReAct chat agent to see the performance

        #  should we return all fetched resources from response object? or rather make another API call to return response + sources

        # TODO 2023-10-05: send the resulting chain of thoughts to gpt3.5 turbo
        # TODO 2023-10-05: update the chain of thought to display each file and chunk used for the reasoning
        # TODO 2023-10-05: return the metadata of each file and chunk used for the reasoning for referencing

    logging.info("Test completed.")
    return retrieval_engine
    # pass
