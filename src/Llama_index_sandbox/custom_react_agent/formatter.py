from llama_index.agent.react.formatter import ReActChatFormatter

from src.Llama_index_sandbox.constants import REACT_CHAT_SYSTEM_HEADER


class CustomReActChatFormatter(ReActChatFormatter):
    """Custom ReAct chat formatter with an updated system header."""

    # Override the system_header attribute with your custom value
    system_header = REACT_CHAT_SYSTEM_HEADER

