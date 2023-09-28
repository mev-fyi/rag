import os
import json
from datetime import datetime
from typing import Dict, Any
import llama_index.response.schema as Response
from llama_index.schema import NodeWithScore

import json
from typing import List, Dict, Any

# Assuming NodeWithScore and Response are imported from the external library


def node_with_score_to_dict(node_with_score: NodeWithScore) -> Dict[str, Any]:
    return {
        "node_id": node_with_score.node_id,
        "score": node_with_score.get_score(),
        "text": node_with_score.text,
        "metadata": node_with_score.metadata
    }


def response_to_dict(response: Response) -> Dict[str, Any]:
    return {
        "response": response.response,
        "source_nodes": [node_with_score_to_dict(node) for node in response.source_nodes],
        "metadata": response.metadata
    }


def store_response(embedding_model_name: str, llm_model_name: str, chunksize: int, chunkoverlap: int, query_str: str, response: Response) -> None:
    # Ensure the directory exists
    dir_path = "datasets/evaluation_results"
    os.makedirs(dir_path, exist_ok=True)

    # Format the filename
    date_str = datetime.now().strftime("%Y-%m-%d")
    file_name = f"{date_str}_{embedding_model_name}_{llm_model_name}_{chunksize}_{chunkoverlap}.json"
    file_path = os.path.join(dir_path, file_name)

    # Convert the response to dict
    data = {
        "subjective_score": "Your subjective score here",
        "query_str": query_str,
        "response": response_to_dict(response),
    }

    # Check if the file already exists
    if os.path.exists(file_path):
        # Load existing data and append the new response
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
        existing_data.append(data)
    else:
        # If the file doesn't exist, initialize the list with the current response
        existing_data = [data]

    # Write back the data to the file
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

# Usage Example:
# embedding_model = "Your embedding model name here"
# llm_model = "Your LLM model name here"
# query_str = "Your query string here"
# response_obj = <Get your Response object from the library>
# store_response(embedding_model, llm_model, query_str, response_obj)
