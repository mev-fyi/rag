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
SYSTEM_MESSAGE = """  
You are an expert in Maximal Extractable Value (MEV)
For any user message that is not related to MEV, blockchain, or mechanism design, or requesting sources about those, respectfully decline to respond and suggest that the user ask a relevant question.
Do not answer based on your prior knowledge. Be exhaustive in your responses and only state facts, do not use hyperboles.
""".strip()

QUERY_TOOL_RESPONSE = """  
The response by the query tool to the question {question} is delimited in the following dash --- delimited block:
---
{response}
---
Based on the question and on the response of the query tool, format a response to the user. If the user requested for sources or content, return the sources provided by the query tool regardless of the query tool response, return these sources. 
""".strip()

LLM_TEMPERATURE = 0.1  # *https://www.youtube.com/watch?v=dW2MmuA1nI4 plays in the background*
OPENAI_MODEL_NAME = "gpt-3.5-turbo-0613"
INPUT_QUERIES = [
        # "What is red teaming in AI",  # Should refuse to respond,
        # "Tell me about LVR",
        # "What plagues current AMM designs?",
        # "How do L2 sequencers work?",
        # "Do an exhaustive breakdown of the MEV supply chain",
        # "What is ePBS?",
        # "What is SUAVE?",
        # "What are intents?",
        "What are the papers that deal with LVR?",
        "What are solutions to mitigate front-running and sandwich attacks?",
        "Give me several sources about L2 sequencers?",
        # "Give me several sources about  L two s sequencers?",
        "Give me several sources about  SUAVE?",
        "Tell me about transaction ordering on L two s",
        "Can you tell me how the definition of MEV evolved over the years?",
        "What are videos that discuss order flow auctions?",
        "Cite all the sources you have about Tim Roughgarden"
    ]

# * Even if it seems like your tools won't be able to answer the question, you must still use them to find the most relevant information and insights. Not using them will appear as if you are not doing your job.
# * You may assume that the users financial questions are related to the documents they've selected.

# The tools at your disposal have access to the following SEC documents that the user has selected to discuss with you:
#     {doc_titles}
# The current date is: {curr_date}


class DOCUMENT_TYPES(Enum):
    YOUTUBE_VIDEO = "youtube_video"
    RESEARCH_PAPER = "research_paper"

