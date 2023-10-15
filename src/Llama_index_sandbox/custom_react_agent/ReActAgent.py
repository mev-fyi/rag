import copy
import json
import logging
from typing import Optional, List

from llama_index.agent import ReActAgent
from llama_index.agent.react.types import BaseReasoningStep
from llama_index.callbacks import trace_method
from llama_index.chat_engine.types import AgentChatResponse
from llama_index.llms import ChatMessage, MessageRole


class CustomReActAgent(ReActAgent):
    from typing import List

    @trace_method("chat")
    def chat(
            self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        """Chat."""
        if chat_history is not None:
            self._memory.set(chat_history)

        self._memory.put(ChatMessage(content=message, role="user"))

        current_reasoning: List[BaseReasoningStep] = []

        # start loop
        for _ in range(self._max_iterations):
            # prepare inputs
            input_chat = self._react_chat_formatter.format(
                chat_history=self._memory.get(), current_reasoning=current_reasoning
            )

            # send prompt
            chat_response = self._llm.chat(input_chat)

            # Create a deep copy of chat_response for modification
            chat_response_copy = copy.deepcopy(chat_response)

            # Enforce user question into Action Input
            response_content = chat_response_copy.raw['choices'][0]['message']['content']
            # NOTE 2023-10-15: we force the input to the query engine to be the user question.
            #  Otherwise, GPT greatly simplifies the question, and the query engine does very poorly.
            if 'Action Input:' in response_content:
                # Extract the part after 'Action Input:'
                action_input_part = response_content.split('Action Input:')[1].strip()

                # Modify its "input" value to be the user question
                try:
                    action_input_json = json.loads(action_input_part)
                    action_input_json['input'] = message

                    # Replace the old part with the modified one
                    response_content = response_content.replace(action_input_part, json.dumps(action_input_json))

                    # Update the deep-copied chat_response accordingly
                    chat_response_copy.raw['choices'][0]['message']['content'] = response_content
                    chat_response_copy.message.content = response_content  # Update this too
                except Exception as e:
                    logging.error(f'Error in modifying the Action Input part of the response_content: [{e}]')

            # given react prompt outputs, call tools or return response
            reasoning_steps, is_done = self._process_actions(output=chat_response_copy)
            current_reasoning.extend(reasoning_steps)

            if is_done:
                break

        response = self._get_response(current_reasoning)
        self._memory.put(
            ChatMessage(content=response.response, role=MessageRole.ASSISTANT)
        )
        return response