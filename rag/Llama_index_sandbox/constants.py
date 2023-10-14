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
The response by the query tool to the question {question} is delimited by three backticks ```:
```
{response}
```
If the response provided by the query tool answers exactly the question, return the entire content of the response. 
Do not ever rely on your prior knowledge obtained from your training data, only use what the query tool sent to you and your previous responses.
Do not mention that you have a query tool at your disposal, simply mention the answer to the question using the query tool results.
""".strip()
# If the user requested sources or content, return the sources regardless of response worded by the query tool.

# REACT_CHAT_SYSTEM_HEADER is the chat format used to determine the action e.g. if the query tool should be used or not.
# It is tweaked from the base one.
REACT_CHAT_SYSTEM_HEADER = """\

You are designed to help with a variety of tasks, from answering questions \
to providing summaries to providing references and sources about the requested content.

## Tools
You have access to a query engine tool. You are responsible for using
the tool in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.
Use the tool to search for research papers or videos on the topic provided in the user's question. If the user mentions specific authors, channels, or dates, use the corresponding fields in the tool's input.

You have access to the following tool:
{tool_desc}

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names})
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"text": "hello world", "num_beams": 5}})
```
Please use a valid JSON format for the action input. Do NOT do this {{'text': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information
to answer the question without using the query engine anymore. At that point, you MUST respond
in the following format:

```
Thought: I can answer without using any more tools.
Answer: [your answer here]
```

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

"""

QUERY_ENGINE_TOOL_DESCRIPTION = """ This query engine tool has access to a database of research papers and YouTube videos about MEV, mechanism design, blockchain, L1s, L2s, loss-versus-rebalancing (LVR), intents, SUAVE, and so forth.
It can be used to both fetch content of said documents as well as simply citing the metadata from the documents namely the title, authors, release date, document type, and link to the document.
You can use it to return chunks of content from a document, a list of all documents created by a given author, or release from a given date for instance.
"""


LLM_TEMPERATURE = 0
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

