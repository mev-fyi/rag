import ast
import json
import re
from typing import Tuple

from llama_index.core.agent.react.output_parser import ReActOutputParser, extract_tool_use
from llama_index.legacy.agent.react.types import BaseReasoningStep, ActionReasoningStep, ObservationReasoningStep, ResponseReasoningStep
from llama_index.legacy.output_parsers.utils import extract_json_str


class CustomReActOutputParser(ReActOutputParser):

    @staticmethod
    def extract_final_response(input_text: str) -> Tuple[str, str]:
        """
        Extracts the "Thought" and "Answer" sections from the provided ReAct agent output.

        Args:
            input_text (str): The output from the ReAct agent.

        Returns:
            Tuple[str, str]: A tuple containing the thought and the answer.

        Raises:
            ValueError: If the expected pattern "Answer:" is not found in the input text.
        """
        # NOTE 2023-10-15: this will be revisited if there are many chain of thoughts with several Observations and Answers
        #  which can be true for expectedly highly complex questions, which have yet to be encountered.

        # Split input text based on "Answer:"
        parts = input_text.split("Answer:", 1)

        # Ensure that there are two parts after the split.
        if len(parts) != 2:
            raise ValueError(f"Could not find the 'Answer:' keyword in input text: {input_text}")

        # Extract thought and answer.
        thought = parts[0].strip()
        answer = parts[1].strip()

        return thought, answer

    def parse(self, output: str, is_streaming=False) -> BaseReasoningStep:
        """Parse output from ReAct agent.

        We expect the output to be in one of the following formats:
        1. If the agent need to use a tool to answer the question:
            ```
            Thought: <thought>
            Action: <action>
            Action Input: <action_input>
            ```
        2. If the agent can answer the question without any tools:
            ```
            Thought: <thought>
            Answer: <answer>
            ```
        """
        if "Thought:" not in output:
            # NOTE: handle the case where the agent directly outputs the answer
            # instead of following the thought-answer format
            return ResponseReasoningStep(
                thought="I can answer without any tools.", response=output
            )

        if "Answer:" in output:
            thought, answer = self.extract_final_response(output)
            return ResponseReasoningStep(thought=thought, response=answer)

        if "Action:" in output:
            thought, action, action_input = extract_tool_use(output)
            json_str = extract_json_str(action_input)

            # First we try json, if this fails we use ast
            try:
                action_input_dict = json.loads(json_str)
            except json.JSONDecodeError:
                action_input_dict = ast.literal_eval(json_str)

            return ActionReasoningStep(
                thought=thought, action=action, action_input=action_input_dict
            )

        raise ValueError(f"Could not parse output: {output}")

