import json
from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload
from typing import Any, Optional, Dict, List


class JSONLoggingHandler(BaseCallbackHandler):

    logs = []

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
            pass
        elif event_type.name == CBEventType.FUNCTION_CALL:
            pass
        elif event_type.name == CBEventType.SYNTHESIZE:
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
        if event_type.name == CBEventType.LLM:
            pass
        elif event_type.name == CBEventType.FUNCTION_CALL:
            pass
        elif event_type.name == CBEventType.SYNTHESIZE:
            pass  # TBD if it is ever hit
        self.logs.append(log_entry)  # or write to a file or logging framework
        # Additional logging logic for the end of the event...

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
