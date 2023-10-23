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
from src.Llama_index_sandbox.constants import NUMBER_OF_CHUNKS_TO_RETRIEVE
from src.Llama_index_sandbox.prompts import QUERY_ENGINE_TOOL_ROUTER

"""
This module contains the JSONLoggingHandler class, a comprehensive logging handler designed for monitoring and recording 
events within a system that interacts with Large Language Model (LLM). 
It captures specific events and data points during the execution flow and logs them in a structured JSON format, allowing for detailed analysis and auditing.

The JSONLoggingHandler extends the BaseCallbackHandler and customizes the event handling process
 for various event types, including but not limited to LLM interactions, function calls, and templating
  events. By overriding methods like `on_event_start` and `on_event_end`, it provides customized logging details for each event type, capturing relevant information depending on the context of the event.

Main Features:
- Custom event logging: Tailors the logging process based on event types, ensuring relevant information is captured accurately for analysis.
- Structured JSON logs: Stores logs in a structured JSON format, making it easier for machines to parse and for humans to read, aiding in debugging and data analysis.
- Real-time logging: Handles events in real-time as they occur within the system, allowing monitoring and potential intervention if necessary.
- Comprehensive event details: Gathers extensive details about events, including payloads, responses, and internal states, offering in-depth insights for audits and performance assessments.

The class manages its log entries in memory for efficient logging and periodically writes to an external JSON file, 
ensuring data persistence while maintaining performance. It is equipped with mechanisms to handle nested logging in specific scenarios, 
such as within function calls, ensuring that the hierarchical relationship between events is preserved in the logs.

Usage:
The JSONLoggingHandler is instantiated with lists of event starts and ends to ignore, 
allowing flexibility in what event types should be excluded from the logs. It is integrated within a larger system and requires proper event emissions compliant with the expected event types and structures.

Note: This class does not implement 'start_trace' and 'end_trace' methods, as tracing may be handled separately or is outside the scope of the current implementation.

Assumptions:
1. Event Structure: The handler assumes that all events emitted in the system adhere to a consistent structure, 
as defined by the `CBEventType` and `EventPayload`. Any deviation in the event structure might lead to logging failures or incorrect log entries.

2. Synchronous Execution: The implementation assumes that events are handled synchronously. 
If the system has asynchronous event handling or concurrent operations, the current logging strategy might miss or incorrectly log events, leading to potential data loss or inconsistency.

3. In-memory Logging: The handler temporarily stores log entries in memory before writing them to a file. 
It assumes that the risk of an application crash is minimal and that such a crash 
won't lead to in-memory log loss. This approach might not be suitable for systems with high fault tolerance requirements.

4. File I/O: The system assumes that file write operations are relatively 
infrequent and inexpensive. In systems where I/O is a bottleneck, frequent logging can exacerbate performance issues.

5. Error Handling: The current implementation has minimal error handling, 
especially for file operations and event parsing. It assumes that events and file operations are mostly error-free, which may not be the case in all environments.

6. Storage Limitations: The system writes logs to a local file, assuming that there is 
always enough disk space available and that logs are managed (e.g., archived, rotated) externally. 
Long-running applications or high-frequency event environments could lead to storage exhaustion.

Limitations and Drawbacks:
1. Scalability: As the volume of events increases, the handler may struggle to keep up with writing logs, 
especially under heavy system load. This limitation is due to the synchronous nature of file writes, which could potentially block or delay event handling.

2. Memory Usage: Keeping all current logs in memory, especially for applications with 
an extensive or complex event flow, might lead to increased memory usage, impacting overall system performance.

3. Lack of Redundancy: The logs are written to a single file without redundancy. Any corruption
 or loss of this file could lead to a complete loss of recent logs, hindering auditing and debugging efforts.

4. No Log Rotation: The handler doesn't implement log rotation or management, which is crucial
 for long-running systems. Without this, log files can grow indefinitely, consuming system resources and making log analysis more challenging.

5. Limited Filtering Capabilities: While the handler allows ignoring certain event starts and ends,
 it doesn't provide a detailed filtering mechanism (e.g., based on event content, source, or other dynamic criteria).
  This limitation could lead to bloated log files containing potentially irrelevant information.

6. Dependency on External Configuration: The handler relies on external setup (like creating the necessary directories for log files).
 In containerized or isolated environments, these dependencies need to be managed externally, adding to deployment complexity.

Developers using or extending this class should be aware of these assumptions and limitations. 
They may need to implement additional safeguards, optimizations, or features, depending on the specific requirements of their system and operational environment.
"""

index_dir = f"{root_dir}/.storage/research_pdf/"
index = sorted(os.listdir(index_dir))[-1].split('_')
index_date = index[0]
embedding_model_name = index[1]
embedding_model_chunk_size = int(index[2])
chunk_overlap = int(index[3])


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
        entry = {}
        if event_type == CBEventType.LLM:
            messages = payload.get(EventPayload.MESSAGES, []) if payload else []
            serialized = payload.get(EventPayload.SERIALIZED, {}) if payload else {}

            if messages[-1].role == MessageRole.USER:
                message_content = messages[-1].content
                if QUERY_ENGINE_TOOL_ROUTER in message_content:
                    user_raw_input = message_content.replace(f"\n{QUERY_ENGINE_TOOL_ROUTER}", "")
                    entry = {
                        "event_type": f"{event_type.name} start",
                        "model_params": serialized,
                        "embedding_model_parameters": {
                            "embedding_model_name": embedding_model_name,
                            "embedding_model_chunk_size": embedding_model_chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "number of chunks to retrieve": NUMBER_OF_CHUNKS_TO_RETRIEVE},
                        "user_raw_input": user_raw_input,
                        "LLM_input": message_content,
                    }
                elif "Context information is below." in message_content:
                    assert TEXT_QA_SYSTEM_PROMPT.content in messages[0].content, "The first message should be the system prompt."
                    tool_output = message_content
                    entry = {
                        "event_type": f"{event_type.name} start",
                        "tool_output": tool_output,
                    }
                else:
                    retrieved_context, previous_answer = self.parse_message_content(message_content)

                    entry = {
                        "event_type": f"{event_type.name} start",
                        "retrieved_context": retrieved_context,
                        "previous_answer": previous_answer,
                    }
            else:
                logging.info(f"WARNING: on_event_start: event_type {event_type.name} was not caught by the logging handler.\n")

        elif event_type == CBEventType.FUNCTION_CALL:
            entry = {"event_type": f"{event_type.name} start", "function_call": []}
            # self.append_to_last_log_entry(function_call)
            self.current_logs.append(entry)
            self.rewrite_log_file()  # Update the log file with the new entry.
            self.current_section = entry["function_call"]
            return

        elif event_type == CBEventType.TEMPLATING:
            if payload:
                template_vars = payload.get(EventPayload.TEMPLATE_VARS, {})
                template = payload.get(EventPayload.TEMPLATE, "")
                entry = {"event_type": f"{event_type.name} start", "instructions": template, "retrieved_chunk": template_vars}
        else:
            logging.info(f"WARNING: on_event_start: event_type {event_type.name} was not caught by the logging handler.\n")

        self.log_entry(entry=entry)

    def log_entry(self, entry):
        """
        Add a new log entry in the current section. If we are within a function call, the entry is nested appropriately.
        """
        if entry.keys():
            if self.current_section is not None:
                # We are inside a nested section, so we should add the log entry here.
                self.current_section.append(entry)
            else:
                # We are not in a nested section, so this entry goes directly under the main log list.
                self.current_logs.append(entry)

        self.rewrite_log_file()  # Update the log file with the new entry.

    def on_event_end(self, event_type: CBEventType, payload: Optional[Dict[str, Any]] = None, event_id: str = "", parent_id: str = "", **kwargs: Any):
        entry = {}

        if event_type == CBEventType.LLM and payload:
            messages = payload.get(EventPayload.MESSAGES, [])
            response = payload.get(EventPayload.RESPONSE, {})
            if response.message.role == MessageRole.ASSISTANT and response.message.content.startswith("Thought: I need to use a tool to help me answer the question."):
                LLM_response = response.message.content
                # self.append_to_last_log_entry({"LLM_response": LLM_response})
                entry = {"event_type": f"{event_type.name} end", "LLM_response": LLM_response}

            elif response.message.role == MessageRole.ASSISTANT and response.message.content.startswith("Thought: I can answer without using any more tools."):
                entry = {"event_type": f"{event_type.name} end", "LLM_response": response.message.content, "subjective grade from 1 to 10": ""}

            elif response.message.role == MessageRole.ASSISTANT:  # catch-all
                entry = {"event_type": f"{event_type.name} end", "LLM_response": response.message.content, "subjective grade from 1 to 10": ""}
            else:
                logging.info(f"WARNING: on_event_end: event_type {event_type.name} was not caught by the logging handler.\n")

        elif event_type == CBEventType.FUNCTION_CALL:
            entry = {"event_type": f"{event_type.name} end", "tool_output": payload.get(EventPayload.FUNCTION_OUTPUT, "")}
            self.current_section = None

        elif event_type == CBEventType.TEMPLATING:
            pass

        else:
            logging.info(f"WARNING: on_event_end: event_type {event_type.name} was not caught by the logging handler.\n")

        self.log_entry(entry=entry)

    def parse_message_content(self, message_content):
        """
        Parse the message content to retrieve 'retrieved_context' and 'previous_answer'.
        This function assumes 'message_content' is a string where the context and answer are
        separated by known delimiter strings.
        """

        # Define your delimiters
        context_start_delim = "New Context:"
        context_end_delim = "Query:"
        answer_start_delim = "Original Answer:"
        answer_end_delim = "New Answer:"
        if "Observation:" in message_content:
            return None, None
        # Find the indices of your delimiters
        try:
            context_start = message_content.index(context_start_delim) + len(context_start_delim)
            context_end = message_content.index(context_end_delim)

            answer_start = message_content.index(answer_start_delim) + len(answer_start_delim)
            answer_end = message_content.index(answer_end_delim)

            # Extract the content based on the indices of the delimiters
            retrieved_context = message_content[context_start:context_end].strip()
            previous_answer = message_content[answer_start:answer_end].strip()
            # Return the extracted information
            return retrieved_context, previous_answer

        except ValueError as e:
            # Handle the case where the delimiters aren't found in the message content
            logging.warning(f"parse_message_content: Error parsing message content: {e}")
            return None, None  # or handle this in a way appropriate for your application

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
