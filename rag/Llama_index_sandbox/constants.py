

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
SYSTEM_MESSAGE = """
You are an expert in Maximal Extractable Value (MEV) that answers questions using the tools at your disposal.
For any user message that is not related to MEV, blockchain, or mechanism design, respectfully decline to respond and suggest that the user ask a relevant question.
""".strip()


LLM_TEMPERATURE = 0.0  # *https://www.youtube.com/watch?v=dW2MmuA1nI4 plays in the background*
OPENAI_MODEL_NAME = "gpt-3.5-turbo-0613"
INPUT_QUERIES = [
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
        "What are videos that discuss order flow auctions?",
        "Cite all the sources you have about Tim Roughgarden"
    ]

# * Even if it seems like your tools won't be able to answer the question, you must still use them to find the most relevant information and insights. Not using them will appear as if you are not doing your job.
# * You may assume that the users financial questions are related to the documents they've selected.

# The tools at your disposal have access to the following SEC documents that the user has selected to discuss with you:
#     {doc_titles}
# The current date is: {curr_date}
