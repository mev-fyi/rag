import os
import dotenv
dotenv.load_dotenv()
from typing import Optional, List
from datetime import datetime

from llama_index.legacy.agent.react.formatter import ReActChatFormatter, get_react_tool_descriptions
from llama_index.legacy.agent.react.types import ObservationReasoningStep, BaseReasoningStep
from llama_index.legacy.core.llms.types import ChatMessage, MessageRole

from src.Llama_index_sandbox.prompts import REACT_CHAT_SYSTEM_HEADER, TWITTER_REACT_CHAT_SYSTEM_HEADER, TOPIC_KEYWORDS


class CustomReActChatFormatter(ReActChatFormatter):
    """Custom ReAct chat formatter with an updated system header."""

    # Override the system_header attribute with your custom value
    is_twitter = True if os.environ.get('TWITTER_BOT', 'FALSE') == 'TRUE' else False
    system_header = REACT_CHAT_SYSTEM_HEADER if not is_twitter else TWITTER_REACT_CHAT_SYSTEM_HEADER

    def format(
        self,
        tools,
        chat_history: List[ChatMessage],
        current_reasoning: Optional[List[BaseReasoningStep]] = None,
    ) -> List[ChatMessage]:
        """Format chat history into list of ChatMessage."""
        current_reasoning = current_reasoning or []

        tool_descs_str = "\n".join(get_react_tool_descriptions(tools))
        current_date = datetime.now().strftime('%Y-%m-%d')
        fmt_sys_header = self.system_header.format(
            current_date=current_date,
            TOPIC_KEYWORDS=TOPIC_KEYWORDS,
            tool_desc=tool_descs_str,
            tool_names=", ".join([tool.metadata.get_name() for tool in tools]),
        )

        # format reasoning history as alternating user and assistant messages
        # where the assistant messages are thoughts and actions and the user
        # messages are observations
        reasoning_history = []
        for reasoning_step in current_reasoning:
            if isinstance(reasoning_step, ObservationReasoningStep):
                message = ChatMessage(
                    role=MessageRole.USER,
                    content=reasoning_step.get_content(),
                )
            else:
                message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=reasoning_step.get_content(),
                )
            reasoning_history.append(message)

        return [
            ChatMessage(role=MessageRole.SYSTEM, content=fmt_sys_header),
            *chat_history,
            *reasoning_history,
        ]