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

from src.Llama_index_sandbox.constants import OPENAI_MODEL_NAME, LLM_TEMPERATURE, SYSTEM_MESSAGE, QUERY_TOOL_RESPONSE, QUERY_ENGINE_TOOL_DESCRIPTION
from src.Llama_index_sandbox.custom_react_agent.ReActAgent import CustomReActAgent
from src.Llama_index_sandbox.custom_react_agent.formatter import CustomReActChatFormatter
from src.Llama_index_sandbox.custom_react_agent.output_parser import CustomReActOutputParser
from src.Llama_index_sandbox.custom_react_agent.tools.fn_schema import ToolFnSchema

from src.Llama_index_sandbox.custom_react_agent.tools.query_engine import CustomQueryEngineTool
from src.Llama_index_sandbox.custom_react_agent.tools.tool_output import log_and_store
from src.Llama_index_sandbox.store_response import store_response
from src.Llama_index_sandbox.utils import timeit


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
                    query_engine_as_tool: bool,
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
    # NOTE 2023-10-14: the description assigned to query_engine_tool should have extra scrutiny as it is passed as is to the agent
    #  and the agent formats it into the react_chat_formatter to determine whether to perform an action with the tool or respond as is.
    # NOTE 2023-10-15: It is unclear how GPT exactly interprets the fn_schema, it is difficult to have a consistent result. Usually GPT greatly
    #  simplifies the query sent to the query engine tool, and the query engine does very poorly. We force the input to the query engine to be the user question.
    query_engine_tool = CustomQueryEngineTool.from_defaults(query_engine=query_engine)
    query_engine_tool.metadata.description = QUERY_ENGINE_TOOL_DESCRIPTION
    query_engine_tool.metadata.fn_schema = ToolFnSchema
    react_chat_formatter: Optional[ReActChatFormatter] = CustomReActChatFormatter(tools=[query_engine_tool])

    # NOTE 2023-10-14: the amount of assumptions baked into the output_parser and how it passes (1) the query to the tool and
    # (2) the final response to be returned to the client, is totally mind-blowing. The simplistic default extract_final_response essentially destroys all content
    output_parser: Optional[ReActOutputParser] = CustomReActOutputParser()
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
    if query_engine_as_tool:
        return CustomReActAgent.from_tools(
            tools=[query_engine_tool],
            react_chat_formatter=react_chat_formatter,
            llm=llm,
            max_iterations=max_iterations,
            memory=memory,
            output_parser=output_parser,
            verbose=verbose,
        )
    else:  # without having query engine as tool (but external to agent)
        return CustomReActAgent.from_tools(
            tools=[],
            react_chat_formatter=react_chat_formatter,
            llm=llm,
            max_iterations=max_iterations,
            memory=memory,
            output_parser=output_parser,
            verbose=verbose,
        )


def ask_questions(input_queries, retrieval_engine, query_engine, store_response_partial, engine, query_engine_as_tool, run_application=False):
    # TODO 2023-10-15: We need metadata filtering at database level else for the query to look over Documents metadata else it fails e.g. when asked to
    #  retrieve content from authors. It would search in paper content but not necessarily correctly fetch all documents, and might return documents that cited the author but which can be irrelevant.
    all_formatted_metadata = None
    for query_str in input_queries:
        # TODO 2023-10-08: add the metadata filters  # https://docs.pinecone.io/docs/metadata-filtering#querying-an-index-with-metadata-filters
        if isinstance(retrieval_engine, BaseChatEngine):
            # TODO 2023-10-07 [RETRIEVAL]: prioritise fetching chunks and metadata from CoT agent
            if not query_engine_as_tool:
                response = query_engine.query(query_str)
                str_response, all_formatted_metadata = log_and_store(store_response_partial, query_str, response, chatbot=True)
                str_response = QUERY_TOOL_RESPONSE.format(question=query_str, response=str_response)
                logging.info(f"Message passed to chat engine:    \n\n[{str_response}]")
                response = retrieval_engine.chat(str_response)
            else:
                logging.info(f"The question asked is: [{query_str}]")
                response = retrieval_engine.chat(query_str)
            if not run_application:
                logging.info(f"[End output shown to client]:    \n```\n{response}\n```")
            # retrieval_engine.reset()

        elif isinstance(retrieval_engine, BaseQueryEngine):
            logging.info(f"Querying index with query:    [{query_str}]")
            response = retrieval_engine.query(query_str)
            response, all_formatted_metadata = log_and_store(store_response_partial, query_str, response, chatbot=False)
        else:
            logging.error(f"Please specify a retrieval engine amongst ['chat', 'query'], current input: {engine}")
            assert False
    if len(input_queries) == 1:
        return response, all_formatted_metadata


@timeit
def get_engine_from_vector_store(embedding_model_name: str,
                                 llm_model_name: str,
                                 chunksize: int,
                                 chunkoverlap: int,
                                 index: VectorStoreIndex,
                                 query_engine_as_tool: bool,
                                 engine='chat',
                                 similarity_top_k=10,
                                 ):

    # TODO 2023-09-29: determine how we should structure our indexes per document type
    service_context: ServiceContext = ServiceContext.from_defaults(llm=OpenAI(model=llm_model_name))
    # create partial store_response with everything but the query_str and response
    store_response_partial = partial(store_response, embedding_model_name, llm_model_name, chunksize, chunkoverlap)

    if engine == 'chat':
        retrieval_engine = get_chat_engine(index=index, service_context=service_context, chat_mode="react", verbose=True, similarity_top_k=similarity_top_k, query_engine_as_tool=query_engine_as_tool)
        retrieval_engine.chat(SYSTEM_MESSAGE)
        query_engine = get_query_engine(index=index, service_context=service_context, verbose=True, similarity_top_k=similarity_top_k)
    elif engine == 'query':
        query_engine = None
        retrieval_engine = get_query_engine(index=index, service_context=service_context, verbose=True, similarity_top_k=similarity_top_k)
    else:
        assert False, f"Please specify a retrieval engine amongst ['chat', 'query'], current input: {engine}"

        # TODO 2023-10-05 [RETRIEVAL]: in particular for chunks from youtube videos, we might want
        #   to expand the window from which it retrieved the chunk
        # TODO 2023-10-05 [RETRIEVAL]: since many chunks can be retrieved from a single youtube video,
        #   what should be the returned timestamp to these references? should we return them all? return the one with highest score?
        # TODO 2023-10-05 [RETRIEVAL]: add weights such that responses from older sources have less importance in the answer
        # TODO 2023-10-05 [RETRIEVAL]: should we weight more a person which is an author and has a paper?
        # TODO 2023-10-07 [RETRIEVAL]: ADD metadata filtering e.g. "only video" or "only papers", or "from this author", or "from this channel", or "from 2022 and 2023" etc
        # TODO 2023-10-07 [RETRIEVAL]: in the chat format, is the rag system keeping in memory the previous retrieved chunks? e.g. if an answer is too short can it develop it further?
        # TODO 2023-10-07 [RETRIEVAL]: should we allow the external user to tune the top-k retrieved chunks? the temperature?

        # TODO 2023-10-09 [RETRIEVAL]: use metadata tags for users to choose amongst LVR, Intents, MEV, etc such that it can increase the result speed (and likely accuracy)
        #  and this upfront work is likely a low hanging fruit relative to payoff.

        #  should we return all fetched resources from response object? or rather make another API call to return response + sources

        # NOTE: 2023-10-05: we can update the chain of thought to display each chunk used for the reasoning

        # TODO 2023-10-15: tweak the Q&A prompt sent to the query engine tool

    return retrieval_engine, query_engine, store_response_partial
