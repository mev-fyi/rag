import json
import logging
from datetime import datetime

from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload
from typing import Any, Optional, Dict, List
import os
from llama_index.llms import MessageRole
from llama_index.prompts.chat_prompts import TEXT_QA_SYSTEM_PROMPT

from src.Llama_index_sandbox import root_dir
from src.Llama_index_sandbox.prompts import QUERY_ENGINE_TOOL_ROUTER


class JSONLoggingHandler(BaseCallbackHandler):

    logs = []

    def __init__(self, event_starts_to_ignore: List[CBEventType], event_ends_to_ignore: List[CBEventType]):
        super().__init__(event_starts_to_ignore, event_ends_to_ignore)

        if not os.path.exists(f"{root_dir}/logs/json"):
            os.makedirs(f"{root_dir}/logs/json")
        self.log_file = f"{root_dir}/logs/json/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log"

        self.current_section = None  # This will point to the part of the log we are currently writing to.
        self.current_logs = []  # Keep all current logs in memory for rewriting.
        with open(self.log_file, 'w') as log:
            json.dump(self.current_logs, log)  # Initialize file with an empty list.

    def on_event_start(self, event_type: CBEventType, payload: Optional[Dict[str, Any]] = None, event_id: str = "", parent_id: str = "", **kwargs: Any):
        # Initial log entry structure
        entry = {}
        if event_type == CBEventType.LLM:
            messages = payload.get(EventPayload.MESSAGES, []) if payload else []
            serialized = payload.get(EventPayload.SERIALIZED, {}) if payload else {}

            if messages[-1].role == MessageRole.USER:  #  len(messages) == 2 and m
                message_content = messages[-1].content
                if QUERY_ENGINE_TOOL_ROUTER in message_content:
                    user_raw_input = message_content.replace(f"\n{QUERY_ENGINE_TOOL_ROUTER}", "")
                    entry = {
                        "event_type": event_type,
                        "model_params": serialized,
                        "user_raw_input": user_raw_input,
                        "LLM_input": message_content,
                    }
                elif "Context information is below." in message_content:
                    # TODO 2023-10-19: make sure this in the array of the function_tool call
                    assert TEXT_QA_SYSTEM_PROMPT.content in messages[0].content, "The first message should be the system prompt."
                    tool_output = message_content
                    entry = {
                        "event_type": event_type,
                        "tool_output": tool_output,
                    }

        elif event_type == CBEventType.FUNCTION_CALL:
            function_call = {"function_call": []}
            self.append_to_last_log_entry(function_call)
            self.current_section = function_call["function_call"]

        elif event_type == CBEventType.TEMPLATING:
            if payload:
                template_vars = payload.get(EventPayload.TEMPLATE_VARS, {})
                template = payload.get(EventPayload.TEMPLATE, "")
                # self.append_to_last_log_entry({"event_type": event_type, "instructions": template, "retrieved_chunk": template_vars})
                entry = {"event_type": event_type, "instructions": template, "retrieved_chunk": template_vars}

        elif event_type == CBEventType.SYNTHESIZE:
            if payload:
                template_vars = payload.get(EventPayload.TEMPLATE_VARS, {})
                template = payload.get(EventPayload.TEMPLATE, "")
                # entry = ({"instructions": template, "retrieved_chunk": template_vars})
                entry = {"event_type": event_type, "instructions": template, "retrieved_chunk": template_vars}
                # self.append_to_last_log_entry({"event_type": event_type, "instructions": template, "retrieved_chunk": template_vars})
        else:
            # log the event that went through and was not caught
            entry = {event_type.name: payload}
            logging.info(f"WARNING: on_event_start: event_type {event_type.name} was not caught by the logging handler.\n"*2)

        self.log_entry(entry=entry)
        # Other event types can be added with elif clauses here...

    def log_entry(self, entry):
        """
        Add a new log entry in the current section. If we are within a function call, the entry is nested appropriately.
        """
        if entry.keys():
            # if self.current_section:
            #     # We're inside a nested section, so the last entry should be here.
            #     last_log_entry = self.current_section[-1]  # No need to check self.current_section again
            # else:
            #     # We're not inside a nested section, so the last entry should be in the main log.
            #     last_log_entry = self.current_logs[-1] if self.current_logs else None
            #
            # # Update the log entry with the LLM response.
            # if last_log_entry is not None:
            #     last_log_entry[entry.keys()[0]] = entry.values()[0]

            if self.current_section is not None:
                # We are inside a nested section, so we should add the log entry here.
                self.current_section.append(entry)
            else:
                # We are not in a nested section, so this entry goes directly under the main log list.
                self.current_logs.append(entry)

        self.rewrite_log_file()  # Update the log file with the new entry.

    def append_to_last_log_entry(self, additional_content):
        """
        Append new content to the last log entry without overwriting existing information.
        This handles both the main log and nested sections.
        """
        if self.current_section is not None:
            # We're inside a nested section, so the last entry should be here.
            target_section = self.current_section
        else:
            # We're not inside a nested section, so the last entry should be in the main log.
            target_section = self.current_logs

        if target_section:
            # Ensure the last log entry is a list where we can append new dictionaries.
            last_log_entry = target_section[-1]

            if isinstance(last_log_entry, list):
                # Append the new content as a separate dictionary within the list.
                last_log_entry.append(additional_content)
            elif isinstance(last_log_entry, dict):
                # If the last entry is a dictionary, we need to decide how to handle it.
                # For example, you could add a new key-value pair where the value is your new content.
                # Here, we're assuming there's a specific key under which content should be added.
                content_key = "additional_content"  # Replace with your actual key.

                # Check if this key already exists and whether its value is a list.
                if content_key in last_log_entry:
                    if isinstance(last_log_entry[content_key], list):
                        # Append the new content to the existing list.
                        last_log_entry[content_key].append(additional_content)
                    else:
                        # If it's not a list, you need to decide how you want to handle it.
                        # You could raise an error, convert it into a list, etc.
                        raise TypeError(f"Expected a list for '{content_key}' but got {type(last_log_entry[content_key])}.")
                else:
                    # If the key doesn't exist, create it and set its value to a list containing your new content.
                    last_log_entry[content_key] = [additional_content]
            else:
                raise TypeError("The last log entry is neither a list nor a dictionary and cannot be appended to.")

            self.rewrite_log_file()  # Update the log file with the new content.
        else:
            # Handle the case where there's no suitable target section to append to.
            raise ValueError("No target section available to append new content.")

    def on_event_end(self, event_type: CBEventType, payload: Optional[Dict[str, Any]] = None, event_id: str = "", parent_id: str = "", **kwargs: Any):
        entry = {}

        if event_type == CBEventType.LLM and payload:
            messages = payload.get(EventPayload.MESSAGES, [])
            response = payload.get(EventPayload.RESPONSE, {})
            if len(messages) == 2 and response.message.role == MessageRole.ASSISTANT:
                LLM_response = response.message.content
                # self.append_to_last_log_entry({"LLM_response": LLM_response})
                entry = {"event_type": event_type, "LLM_response": LLM_response}

        elif event_type == CBEventType.FUNCTION_CALL:
            self.current_section = None

        elif event_type.name == CBEventType.SYNTHESIZE:
            pass  # TBD if it is ever hit

        else:
            # log the event that went through and was not caught
            entry = {event_type.name: payload}
            logging.info(f"WARNING: on_event_end: event_type {event_type.name} was not caught by the logging handler.\n"*2)

        self.log_entry(entry=entry)

    def rewrite_log_file(self):
        # A helper method to handle writing the logs to the file.
        with open(self.log_file, 'w') as log:  # Note the 'w' here; we're overwriting the file.
            json.dump(self.current_logs, log, indent=4)  # Pretty-print for readability.

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Not implemented."""

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Not implemented."""
