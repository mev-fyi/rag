from enum import Enum


# SYSTEM_MESSAGE = """
# You are an expert in Maximal Extractable Value (MEV) that answers questions using the tools at your disposal.
# These tools have information regarding MEV research including academic papers, articles, diarized transcripts from conversations registered on talks at conferences or podcasts.
# Here are some guidelines that you must follow:
# * For any user message that is not related to MEV, blockchain, or mechanism design, respectfully decline to respond and suggest that the user ask a relevant question.
# * If your tools are unable to find an answer, you should say that you haven't found an answer.
#
# Now answer the following question:
# {question}
# """.strip()

#  that answers questions using the query tools at your disposal.

# If the user requested sources or content, return the sources regardless of response worded by the query tool.

# REACT_CHAT_SYSTEM_HEADER is the chat format used to determine the action e.g. if the query tool should be used or not.
# It is tweaked from the base one.

# You are designed to help with a variety of tasks, from answering questions \
# to providing summaries to providing references and sources about the requested content.


# You are responsible for using
# the tool in any sequence you deem appropriate to complete the task at hand.
# This may require breaking the task into subtasks and using different tools
# to complete each subtask.


LLM_TEMPERATURE = 0
NUMBER_OF_CHUNKS_TO_RETRIEVE = 10

'''
valid OpenAI model name in: gpt-4, gpt-4-32k, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0613, 
gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, 
text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo
'''

OPENAI_MODEL_NAME = "gpt-3.5-turbo-0613"  # "gpt-4-0613"  # "gpt-3.5-turbo-0613"
INPUT_QUERIES = [
        # "What is red teaming in AI",  # Should refuse to respond,
        "Tell me about LVR",
        # "What plagues current AMM designs?",
        "How do L2 sequencers work?",
        "Do an exhaustive breakdown of the MEV supply chain",
        "What is ePBS?",
        "What is SUAVE?",
        "Give me the most exhaustive definition of loss-versus-rebalancing (LVR)",
        "What are intents?",
        "What are the papers that deal with LVR?",
        "What are solutions to mitigate front-running and sandwich attacks?",
        "Give me several sources about L2 sequencing?",
        "Give me several sources about SUAVE?",
        "Tell me about transaction ordering on L2s",
        # "Can you tell me how the definition of MEV evolved over the years?",
        # "What are videos that discuss order flow auctions?",
        # "Cite all the sources you have about Tim Roughgarden"
    ]


EVALUATION_INPUT_QUERIES = [
        "Tell me about LVR",  #1
        "How do L2 sequencers work?",  #2
        "Do an exhaustive breakdown of the MEV supply chain",  #3
        "What is ePBS?",  #4
        "What is SUAVE?",  #5
        "Give me the most exhaustive definition of loss-versus-rebalancing (LVR)",  #6
        "What are intents?",  #7
        "What are the papers that deal with LVR?",  #8
        "What are solutions to mitigate front-running and sandwich attacks?",  #9
        "Give me several sources about L2 sequencing?",  #10
        "Give me several sources about SUAVE?",  #11
        "Tell me about transaction ordering on L2s",  #12
    ]

# * Even if it seems like your tools won't be able to answer the question, you must still use them to find the most relevant information and insights. Not using them will appear as if you are not doing your job.
# * You may assume that the users financial questions are related to the documents they've selected.

# The tools at your disposal have access to the following SEC documents that the user has selected to discuss with you:
#     {doc_titles}
# The current date is: {curr_date}


class DOCUMENT_TYPES(Enum):
    YOUTUBE_VIDEO = "youtube_video"
    RESEARCH_PAPER = "research_paper"

