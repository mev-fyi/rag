from enum import Enum

from llama_index.legacy.callbacks import EventPayload


# Enums are not designed to be extended or altered once defined. They are a fixed set of constants, and Python enforces their immutability.

# Define a new Enum that includes all the members of EventPayload, plus your additions
class ExtendedEventPayload(str, Enum):
    # These are the original members, repeated here
    DOCUMENTS = EventPayload.DOCUMENTS.value
    CHUNKS = EventPayload.CHUNKS.value
    NODES = EventPayload.NODES.value
    PROMPT = EventPayload.PROMPT.value
    MESSAGES = EventPayload.MESSAGES.value
    COMPLETION = EventPayload.COMPLETION.value
    RESPONSE = EventPayload.RESPONSE.value
    QUERY_STR = EventPayload.QUERY_STR.value
    SUB_QUESTION = EventPayload.SUB_QUESTION.value
    EMBEDDINGS = EventPayload.EMBEDDINGS.value
    TOP_K = EventPayload.TOP_K.value
    ADDITIONAL_KWARGS = EventPayload.ADDITIONAL_KWARGS.value
    SERIALIZED = EventPayload.SERIALIZED.value
    FUNCTION_CALL = EventPayload.FUNCTION_CALL.value
    FUNCTION_OUTPUT = EventPayload.FUNCTION_OUTPUT.value
    TOOL = EventPayload.TOOL.value
    MODEL_NAME = EventPayload.MODEL_NAME.value
    TEMPLATE = EventPayload.TEMPLATE.value
    TEMPLATE_VARS = EventPayload.TEMPLATE_VARS.value
    SYSTEM_PROMPT = EventPayload.SYSTEM_PROMPT.value
    QUERY_WRAPPER_PROMPT = EventPayload.QUERY_WRAPPER_PROMPT.value
    EXCEPTION = EventPayload.EXCEPTION.value

    FORMATTED_METADATA = "formatted_metadata"

