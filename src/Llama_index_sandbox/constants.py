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
TEXT_SPLITTER_CHUNK_SIZE = 700
TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE = 10

'''
valid OpenAI model name in: gpt-4, gpt-4-32k, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0613, 
gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, 
text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo
'''
OPENAI_INFERENCE_MODELS = ["gpt-4", "gpt-4-32k", "gpt-4-0613", "gpt-4-32k-0613", "gpt-4-0314", "gpt-4-32k-0314", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613",
"gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0301", "text-davinci-003", "text-davinci-002", "gpt-3.5-turbo-instruct", "gpt-35-turbo-16k", "gpt-35-turbo", "gpt-4-1106-preview", "gpt-4-turbo", "gpt-4-turbo-1106"]


OPENAI_MODEL_NAME = "gpt-3.5-turbo"  #"gpt-3.5-turbo-16k-0613"  # "gpt-3.5-turbo-0613"  # "gpt-4" # "gpt-3.5-turbo-0613"  # "gpt-4-0613"  # "gpt-3.5-turbo-0613"
INPUT_QUERIES = [
        # "What are FRPs?",
        # "Tell me about LVR",  # 1
        # "should I be concerned about it?",
        # "How do L2 sequencers work?",  # 2
        # "Do an exhaustive breakdown of the MEV supply chain",  # 3
        # "What is ePBS?",  # 4
        "What is SUAVE?",  # 5
        "Give me the most exhaustive definition of loss-versus-rebalancing (LVR)",  # 6
        "What are intents?",  # 7
        "What is ePBS?",
        "What's PEPC?",
        "What are the papers that deal with LVR?",  # 8
        "What are solutions to mitigate front-running and sandwich attacks?",  # 9
        "Give me several sources about L2 sequencing?",  # 10
        "Give me several sources about SUAVE?",  # 11
        "Tell me about transaction ordering on L2s",  # 12
        "What are OFAs?",
        "Can you tell me how the definition of MEV evolved over the years?",
        "What is MEV burn?",
        "What is account abstraction?",
        "What is 4337?",
        "What is 1559?",
        "Who are the most active individuals in the MEV space?",
        "How will account abstraction affect the MEV supply chain?",
        "What is the difference between account abstraction and intents?",
        "Is it bad that MEV is a centralizing force?",
        "How is MEV on Solana different from MEV on Ethereum?",
        "Explain to me the differences between Uniswap v2, v3 and v4",
        "What are commit/reveal schemes?",
        "What is the impact of latency in MEV?",
        "What is PEPC?",
        "Are roll-ups real?",
        "Are intents real?",
        "What are relays?",
        "How does MEV compare across chains for instance Ethereum, Solana, Arbitrum?",
        "What are payment for order flows in MEV?",
        "What is Anoma?",
        "What are shared sequencers?",
        "What is TEE?",
        "What is MPC?",
        "How does TEE and MPC relate to MEV?",
        "What is Ethereum alignment?",
        "Return a selection of papers and videos that will introduce me to MEV",
        "What is the role of the Ethereum Foundation?",
        "How does Flashbots contribute to the MEV space?",
        "Give me talks from Barnabe Monnot",
        "Given your knowledge of transaction ordering, market microstructure and design, what Uniswap V4 hook designs would you recommend to mitigate LVR?",
        "What is atomic composability?",
        "What are the main advantages and challenges that decentralised finance face relative to traditional finance?",
        "What is the number one thing which make decentralised finance better than traditional finance and why?",
        """The Limits of Atomic Composability and Shared Sequencing
        Problem statement: Perhaps the most-talked-about path to scaling blockchains today is deploying many rollups on Ethereum. While this bootstraps security and decentralization, deploying many disconnected rollups on Ethereum will fracture composability. We believe atomic composability – the ability to send a transaction A that finalizes if, and only if, transaction B is finalized – is crucial.
        Please describe the limits of composability between rollups on Ethereum. Ideally a solution would propose a formal model of rollups and an impossibility result. """,
        """Optimal LVR Mitigation
        Problem Statement:
        Loss vs. rebalancing (aka LVR and pronounced ‘Lever’) was proposed in a 2022 paper as a way of modeling adverse selection costs borne
        by liquidity providers to constant function market maker decentralized exchanges (CFMM DEXs). Current work is focused on finding an optimal way to mitigate LVR in DEXs without using a price oracle.
        Please describe the potential mitigations to LVR and argue why your proposed solution is better than all known alternatives. """,
        """Designing the MEV Transaction Supply Chain
        Problem Statement:
        Assuming you could start from scratch, what is the optimal design of the miner extractable value (MEV) transaction supply chain? The process today is most naturally separated into distinct roles for searchers, builders, and proposers. What are the economic tradeoffs for maintaining these as separate roles versus having them consolidate? Are there new roles that would be beneficial to introduce? What are the optimal mechanisms to mediate how these different parties interact? Can the mechanisms mediating how the MEV supply chain functions be purely economic or are there components that require cryptographic solutions/trusted components?
        The notion of what “optimal” means is intentionally left vague. Argue for what metrics are the most important when evaluating different mechanisms. Do we require strict collusion resistance between any groups of agents throughout the supply chain? Do we only require collusion resistance between agents at the same level of the supply chain? Is it enough that the mechanism’s properties hold in equilibrium or is it important that all parties have dominant strategies? On the other hand, what are lower bounds for how “optimal” the transaction supply chain can be? Are there certain conditions under which it is impossible to achieve all the “optimal” properties we might want?
        This problem is left open to interpretation. Feel free to address any of the questions above or provide your own direction towards designing mechanisms for the transaction supply chain. """,
        "What is referred to as good versus bad MEV? How would you explain that to a layman?",
        "What are the consensus trade-offs that a protocol must make between MEV and decentralization?",
        "What are the consensus trade-offs that a protocol must make to obtain a higher transaction throughput?",
        "What are credible commitments?",
        "What is at the intersection of AI and crypto?",
        "Would a spot ETH ETF be good for the crypto ecosystem? Would that be a centralising force?",
        "How can auction theory be used to design a better MEV auction?",
        "What are all the subjects that are needed to understand MEV?",
        "What are all the subjects you are an expert in?",
        "What is MEV-Share?",
        "What is MEV-Boost?",
        "What is a builder?",
        "What is a searcher?",
        "What is a validator?",
        "What is an attester?",
        "What is an integrated searcher builder?",
        "How do searchers, builders, relays and validators interact with one another?",
        "How can the user initiating a transaction get shielded from MEV?",
        "Who has most power in the MEV supply chain across users, wallets, searchers, builders, relays, validators?",
]

EVALUATION_INPUT_QUERIES = [
        "Tell me about LVR",
        "How do L2 sequencers work?",
        "Do an exhaustive breakdown of the MEV supply chain",
        "What is ePBS?",
        "What is SUAVE?",
        "Give me the most exhaustive definition of loss-versus-rebalancing (LVR)",
        "What are intents?",
        "What are the papers that deal with LVR?",
        "What are solutions to mitigate front-running and sandwich attacks?",
        "Give me several sources about L2 sequencing",
        "Give me several sources about SUAVE?",
        "Tell me about transaction ordering on L2s",
        "What are OFAs?",
        "Can you tell me how the definition of MEV evolved over the years?",
        "What is MEV burn?",
        "What is account abstraction?",
        "What is 4337?",
        "What is 1559?",
        "Who are the most active individuals in the MEV space?",
        "How will account abstraction affect the MEV supply chain?",
        "What is the difference between account abstraction and intents?",
        "Is it bad that MEV is a centralizing force?",
        "How is MEV on Solana different from MEV on Ethereum?",
        "Explain to me the differences between Uniswap v2, v3 and v4",
        "What are commit/reveal schemes?",
        "What is the impact of latency in MEV?",
        "What is PEPC?",
        "Are roll-ups real?",
        "Are intents real?",
        "What are relays?",
        "How does MEV compare across chains for instance Ethereum, Solana, Arbitrum?",
        "What are payment for order flows in MEV?",
        "What is Anoma?",
        "What are shared sequencers?",
        "What is TEE?",
        "What is MPC?",
        "How does TEE and MPC relate to MEV?",
        "What is Ethereum alignment?",
        "Return a selection of papers and videos that will introduce me to MEV",
        "What is the role of the Ethereum Foundation?",
        "How does Flashbots contribute to the MEV space?",
        "Give me talks from Barnabe Monnot",
        "Given your knowledge of transaction ordering, market microstructure and design, what Uniswap V4 hook designs would you recommend to mitigate LVR?",
        "What is atomic composability?",
        "What are the main advantages and challenges that decentralised finance face relative to traditional finance?",
        "What is the number one thing which make decentralised finance better than traditional finance and why?",
        """The Limits of Atomic Composability and Shared Sequencing
        Problem statement: Perhaps the most-talked-about path to scaling blockchains today is deploying many rollups on Ethereum. While this bootstraps security and decentralization, deploying many disconnected rollups on Ethereum will fracture composability. We believe atomic composability – the ability to send a transaction A that finalizes if, and only if, transaction B is finalized – is crucial. 
        Please describe the limits of composability between rollups on Ethereum. Ideally a solution would propose a formal model of rollups and an impossibility result. """,
        """Optimal LVR Mitigation
        Problem Statement:
        Loss vs. rebalancing (aka LVR and pronounced ‘Lever’) was proposed in a 2022 paper as a way of modeling adverse selection costs borne 
        by liquidity providers to constant function market maker decentralized exchanges (CFMM DEXs). Current work is focused on finding an optimal way to mitigate LVR in DEXs without using a price oracle. 
        Please describe the potential mitigations to LVR and argue why your proposed solution is better than all known alternatives. """,
        """Designing the MEV Transaction Supply Chain
        Problem Statement: 
        Assuming you could start from scratch, what is the optimal design of the miner extractable value (MEV) transaction supply chain? The process today is most naturally separated into distinct roles for searchers, builders, and proposers. What are the economic tradeoffs for maintaining these as separate roles versus having them consolidate? Are there new roles that would be beneficial to introduce? What are the optimal mechanisms to mediate how these different parties interact? Can the mechanisms mediating how the MEV supply chain functions be purely economic or are there components that require cryptographic solutions/trusted components?
        The notion of what “optimal” means is intentionally left vague. Argue for what metrics are the most important when evaluating different mechanisms. Do we require strict collusion resistance between any groups of agents throughout the supply chain? Do we only require collusion resistance between agents at the same level of the supply chain? Is it enough that the mechanism’s properties hold in equilibrium or is it important that all parties have dominant strategies? On the other hand, what are lower bounds for how “optimal” the transaction supply chain can be? Are there certain conditions under which it is impossible to achieve all the “optimal” properties we might want?
        This problem is left open to interpretation. Feel free to address any of the questions above or provide your own direction towards designing mechanisms for the transaction supply chain. """,
        "What is referred to as good versus bad MEV? How would you explain that to a layman?",
        "What are the consensus trade-offs that a protocol must make between MEV and decentralization?",
        "What are the consensus trade-offs that a protocol must make to obtain a higher transaction throughput?",
        "What are credible commitments?",
        "What is at the intersection of AI and crypto?",
        "Would a spot ETH ETF be good for the crypto ecosystem? Would that be a centralising force?",
        "How can auction theory be used to design a better MEV auction?",
        "What are all the subjects that are needed to understand MEV?",
        "What are all the subjects you are an expert in?",
        "What is MEV-Share?",
        "What is MEV-Boost?",
        "What is a builder?",
        "What is a searcher?",
        "What is a validator?",
        "What is an attester?",
        "What is an integrated searcher builder?",
        "How do searchers, builders, relays and validators interact with one another?",
        "How can the user initiating a transaction get shielded from MEV?",
        "Who has most power in the MEV supply chain across users, wallets, searchers, builders, relays, validators?",
]


# * Even if it seems like your tools won't be able to answer the question, you must still use them to find the most relevant information and insights. Not using them will appear as if you are not doing your job.
# * You may assume that the users financial questions are related to the documents they've selected.

# The tools at your disposal have access to the following SEC documents that the user has selected to discuss with you:
#     {doc_titles}
# The current date is: {curr_date}


class DOCUMENT_TYPES(Enum):
    YOUTUBE_VIDEO = "youtube_video"
    RESEARCH_PAPER = "research_paper"
    ARTICLE = "article"

