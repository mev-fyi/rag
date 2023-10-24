# Constants representing the search matrix parameters. They are defined in all caps to follow the convention for constants.

# TODO 2023-10-24: another variable to test for would be the data ingestion e.g. what PDF reader do we use or how often do we renew newlines in YouTube transcripts
#

# Assumption: the chunk sizes and overlaps are the same regardless of PDF or youtube video
# Embedding model sizes as a percentage of the maximum size.
CHUNK_SIZES = [20, 50, 70, 100]  # Representing percentages of EMBEDDING_DIMENSIONS from config.

# Overlapping of chunks, represented as a percentage.
CHUNK_OVERLAPS = [5, 10, 15]  # Representing percentages.

# Number of chunks retrieved for analysis.
NUM_CHUNKS_RETRIEVED = [3, 5, 7, 10]

# Different embedding models ordered from expected best to worst performance.
EMBEDDING_MODELS = [
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "llmrails/ember-v1",
    "thenlper/gte-large",
    "text-embedding-ada-002",
]

INFERENCE_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-13b-chat-hf"
    "meta-llama/Llama-2-70b-chat-hf"
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4-0613",
    # Optional: Include any fine-tuned models here. For example:
    # "custom-fine-tuned-gpt-3.5"
]


# Now, these lists are set up for use in further analysis or a parameterized experiment setup.
# You can access these lists and iterate over them to set up your experiments or analyses,
# potentially using nested loops to cover every combination of parameters.
