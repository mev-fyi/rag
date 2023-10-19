import json
from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload
from typing import Any, Optional, Dict, List


class JSONLoggingHandler(BaseCallbackHandler):

    logs = []

    def __init__(self, event_starts_to_ignore: List[CBEventType], event_ends_to_ignore: List[CBEventType]):

        super().__init__(event_starts_to_ignore, event_ends_to_ignore)
        self.log_file = "path_to_your_log_file.log"  # specify your log file path
        self.indentation_level = 0



    def on_event_start(self,
                       event_type: CBEventType,
                       payload: Optional[Dict[str, Any]] = None,
                       event_id: str = "",
                       parent_id: str = "",
                       **kwargs: Any):
        log_entry = {
            'event_id': event_id,
            'parent_id': parent_id,
            'event_type': event_type.name,  # assuming event_type is an Enum
            'status': 'started',
            'payload': payload,
        }
        # TODO 2023-10-17: impl for each event_type
        # LLM is level 1
        # Successive Function Calls are nested under LLM. It seems at the first glance that these successive calls (get_responsive_refine)do not trigger events of their own
        # Then un-indented
        # if the length last message in EventPayload.MESSAGES (which is a list of CHatMessage objects) starts with "user:" then start a new json array and write that message to the first index
        # then write as a single entry the EventPayload.SERIALIZED which is a dict. It contains model, temperature, max_retries etc.

        if event_type.name == CBEventType.LLM:
            messages = payload.get(EventPayload.MESSAGES, [])
            serialized = payload.get(EventPayload.SERIALIZED, {})
            # if the length of EventPayload.MESSAGES is two, and that the last message starts with "user: Context information is below" then it is a function call from the query tool
            # and any log entry should be nested under the topmost LLM event / at the same level of the CBEventType.FUNCTION_CALL call
            if len(messages) == 2 and messages[-1].startswith("user:"):
                # Here you'd implement whatever specific logic you need for this condition.
                # For example, starting a new JSON array in the log or handling the message differently.
                self.start_new_log_section(messages[-1])
                self.log_entry(serialized)  # This is assumed to be a method that handles logging of entries
            if len(messages) == 2 and messages[-1].startswith("user: Context information is below"):
                # log as usual
                pass
        elif event_type.name == CBEventType.FUNCTION_CALL:
            # when this event hit, you will need to indent every coming json file until the CBEventType.FUNCTION_CALL ends
            # Specific logic for function call events, such as nesting logs, can be implemented here.
            # If you need to indent JSON, you might adjust how you're serializing the log entries.
            self.indent_logs()
            pass
        elif event_type.name == CBEventType.TEMPLATING:
            # when this is hit, then the EventPayload.TEMPLATE_VARS  contains a dict of the retrieved chunk, "context_msg" which maps to the retrieved chunk.
            # I need you to save that chunk in a variable and then write it to the json file, still at the same indentation level of the function_call.
            # also write the EventPayload.TEMPLATE (str) which contains the instructions, right above the retrieved chunk
            pass
        elif event_type.name == CBEventType.SYNTHESIZE:
            template_vars = payload.get(EventPayload.TEMPLATE_VARS, {})
            template = payload.get(EventPayload.TEMPLATE, "")

            # Write specific parts of the payload to the log, as per the comments.
            self.log_entry({"instructions": template, "retrieved_chunk": template_vars})
            pass  # TBD if it is ever hit
        self.logs.append(log_entry)  # or write to a file or logging framework
        # Additional logging logic for the start of the event...

    def on_event_end(self,
                     event_type: CBEventType,
                     payload: Optional[Dict[str, Any]] = None,
                     event_id: str = "",
                     parent_id: str = "",
                     **kwargs: Any):
        log_entry = {
            'event_id': event_id,
            'event_type': event_type.name,  # assuming event_type is an Enum
            'status': 'ended',
            'payload': payload,
        }
        if event_type.name == CBEventType.LLM and payload:
            # if the length of EventPayload.MESSAGES is two, and that the last message starts with "assistant:", and that the last log then it is a function call from the query tool
            messages = payload.get(EventPayload.MESSAGES, [])

            # Condition check for specific scenario in LLM event end
            if len(messages) == 2 and messages[-1].startswith("assistant:"):
                # Specific logic for this condition
                pass  # Your implementation here
            pass
        elif event_type.name == CBEventType.FUNCTION_CALL:
            # If there was some sort of nesting or structuring started in on_event_start, we'd close or resolve it here.
            self.outdent_logs()
            pass
        elif event_type.name == CBEventType.SYNTHESIZE:
            pass  # TBD if it is ever hit
        self.logs.append(log_entry)  # or write to a file or logging framework
        # Additional logging logic for the end of the event...

    def write_to_log(self, data):
        with open(self.log_file, 'a') as log:
            # We're writing JSON strings for readability and structure
            log.write(json.dumps(data, indent=4) + "\n")  # json.dumps serializes the data to a formatted string

    def start_new_log_section(self, message):
        # This method will write a new section header in the log file
        section_header = {
            "section_start": message,
            "content": []
        }
        self.write_to_log(section_header)

    def log_entry(self, entry):
        # Here, we're adding a standard log entry
        structured_entry = {
            "log_entry": entry,
            "indentation": self.indentation_level
        }
        self.write_to_log(structured_entry)

    def indent_logs(self):
        # Increase the indentation level for nested structures
        self.indentation_level += 1

    def outdent_logs(self):
        # Decrease the indentation level after ending a nested structure
        if self.indentation_level > 0:
            self.indentation_level -= 1

    def get_logs(self):
        return json.dumps(self.logs, indent=4)  # For pretty printing

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Not implemented."""

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Not implemented."""


# Simulating chat object handling
def handle_chat_object(chat_object):
    # Assuming 'chat_object' contains the data you want to log

    metadata = extract_metadata(chat_object)  # Your method to extract metadata
    response = extract_response(chat_object)  # Your method to extract response

    # Create a context for the chat event
    with callback_manager.event(CBEventType.CHAT_OBJECT) as event:
        # You can include more detailed payloads as needed
        event.on_start(payload={"metadata": metadata})

        # ... (process the chat object) ...

        event.on_end(payload={"response": response})


# After processing you can get the logs
