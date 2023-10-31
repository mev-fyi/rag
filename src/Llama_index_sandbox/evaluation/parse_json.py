import json
import os

from src.Llama_index_sandbox import root_directory, root_dir


def parse_old_json_format():
    # Define the path to your original file. Please make sure this path is correct.
    original_file_path = f'{root_directory()}/datasets/golden_source_logs/2023-10-20_14:32:07.log.json'
    # Define the path for your new subdirectory.
    subdirectory_path = f'{root_directory()}/datasets/golden_source_logs/parsed_jsons'

    # Check if the subdirectory does not exist.
    if not os.path.exists(subdirectory_path):
        # Create the subdirectory.
        os.makedirs(subdirectory_path)

    # Define the path to your new file within the subdirectory.
    new_file_path = os.path.join(subdirectory_path, '2023-10-20_14:32:07.log.json')

    # Initialize the list that will hold all results.
    results = []

    # Try to open your file and parse it as JSON.
    try:
        with open(original_file_path, 'r', encoding='utf-8') as file:
            json_list = json.load(file)

        model_params = None
        user_raw_input = None
        final_answer = None

        for idx, event in enumerate(json_list):
            if event["event_type"] == "llm start":
                # If we're not currently tracking a question, start a new one.
                if model_params is None:
                    model_params = event.get("model_params", {})
                if user_raw_input is None:
                    user_raw_input = event.get("user_raw_input", "")

            elif event["event_type"] == "llm end" and model_params and user_raw_input:
                # Check the response starts with the specific string.
                condition_1 = event.get("LLM_response", "").startswith("Thought: I can answer without using any more tools.\nAnswer:")
                condition_2 = False  # Initializing the variable

                # Check that we're not at the last index to avoid 'index out of range' errors.
                if idx < len(json_list) - 1:
                    next_event = json_list[idx + 1]
                    next_event_type = next_event["event_type"]
                    next_event_model_params = next_event.get("model_params", None)
                    condition_2 = next_event_type == "llm start" and next_event_model_params is not None

                if condition_1 or condition_2:
                    # If condition_1 is met, we need to split the response to extract the final answer.
                    # If condition_2 is met, it implies the answer didn't start with the specific string, so we take the entire response.
                    final_answer = event.get("LLM_response", "").split("Answer: ")[1].strip() if condition_1 else event.get("LLM_response", "")

                    # Save the gathered information.
                    results.append({
                        "model_params": model_params,
                        "user_raw_input": user_raw_input,
                        "LLM_response": final_answer
                    })

                    # Reset for the next pair.
                    model_params = None
                    user_raw_input = None
                    final_answer = None

        # After processing all events, check the results and write them to a new file.
        with open(new_file_path, 'w', encoding='utf-8') as outfile:
            if results:
                json.dump(results, outfile, indent=4)
                print(f"Results were written to {new_file_path}")
            else:
                print("No valid event pairs were found in the logs. No file was written.")

    except FileNotFoundError:
        print(f"The file {original_file_path} does not exist.")
    except json.JSONDecodeError:
        print("The file doesn't contain valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_latest_raw_json_log():
    """
    Fetch the most recent file from a specific directory.

    :param directory: str, The directory path where the files are stored.
    :return: str, The file name of the most recent file.
    """
    directory = f"{root_dir}/logs/json"
    # Check if the directory exists.
    if not os.path.exists(directory):
        print(f"Directory: {directory} does not exist.")
        return None

    # Get all the files in the directory.
    full_file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

    # Check if the directory is empty.
    if not full_file_paths:
        print(f"No files found in the directory: {directory}")
        return None

    # Get the latest file.
    latest_file = max(full_file_paths, key=os.path.getctime)  # Change to os.path.getmtime if considering the last modification time.

    return latest_file


def get_latest_parsed_json_log():
    """
    Fetch the most recent file from a specific directory.

    :param directory: str, The directory path where the files are stored.
    :return: str, The file name of the most recent file.
    """
    directory = f"{root_dir}/datasets/golden_source_logs/parsed_jsons"
    # Check if the directory exists.
    if not os.path.exists(directory):
        print(f"Directory: {directory} does not exist.")
        return None

    # Get all the files in the directory.
    full_file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

    # Check if the directory is empty.
    if not full_file_paths:
        print(f"No files found in the directory: {directory}")
        return None

    # Get the latest file.
    latest_file = max(full_file_paths, key=os.path.getctime)  # Change to os.path.getmtime if considering the last modification time.

    return latest_file


def parse_2023_10_24_json_format(raw_json_log_path=None):
    # Define the path to your original file. Please make sure this path is correct.
    if raw_json_log_path is None:
        original_file_path = get_latest_raw_json_log()
    else:
        original_file_path = raw_json_log_path
    file_name = original_file_path.split('/')[-1]
    if file_name:
        print(f"The most recent file is: {file_name}")
    else:
        print("Could not retrieve the file.")

    # Define the path for your new subdirectory.
    subdirectory_path = f'{root_directory()}/datasets/golden_source_logs/parsed_jsons'

    # Check if the subdirectory does not exist.
    if not os.path.exists(subdirectory_path):
        # Create the subdirectory.
        os.makedirs(subdirectory_path)

    # Define the path to your new file within the subdirectory.
    new_file_path = os.path.join(subdirectory_path, file_name)

    # Initialize the list that will hold all results.
    results = []

    # Try to open your file and parse it as JSON.
    try:
        with open(original_file_path, 'r', encoding='utf-8') as file:
            json_list = json.load(file)

        model_params = None
        embedding_model_parameters = None
        user_raw_input = None
        final_answer = None

        for idx, event in enumerate(json_list):
            if event["event_type"].lower() == "llm start":
                # If we're not currently tracking a question, start a new one.
                if model_params is None:
                    model_params = event.get("model_params", {})
                if embedding_model_parameters is None:
                    embedding_model_parameters = event.get("embedding_model_parameters", {})
                if user_raw_input is None:
                    user_raw_input = event.get("user_raw_input", "")

            elif event["event_type"].lower() == "llm end" and model_params and user_raw_input and embedding_model_parameters:
                # Check the response starts with the specific string.
                condition_1 = event.get("LLM_response", "").startswith("Thought: I can answer without using any more tools.\nAnswer:")
                condition_2 = False  # Initializing the variable

                # NOTE 2023-10-24: this is hacky but it works for now as opposed to writing the metadata in the LLM end payload.
                prev_two_event = json_list[idx - 2] if idx > 2 else None
                prev_two_event_type = prev_two_event["event_type"].lower() if prev_two_event else None
                metadata = None
                if prev_two_event_type == 'FUNCTION_CALL end'.lower():
                    metadata = prev_two_event.get("metadata", None)

                # Check that we're not at the last index to avoid 'index out of range' errors.
                if idx < len(json_list) - 1:
                    next_event = json_list[idx + 1]
                    next_event_type = next_event["event_type"].lower()
                    next_event_model_params = next_event.get("model_params", None)
                    condition_2 = next_event_type == "llm start" and next_event_model_params is not None

                if condition_1 or condition_2:
                    # If condition_1 is met, we need to split the response to extract the final answer.
                    # If condition_2 is met, it implies the answer didn't start with the specific string, so we take the entire response.
                    final_answer = event.get("LLM_response", "").split("Answer: ")[1].strip() if condition_1 else event.get("LLM_response", "")

                    # Save the gathered information.
                    results.append({
                        "model_params": model_params,
                        "embedding_model_parameters": embedding_model_parameters,
                        "user_raw_input": user_raw_input,
                        "LLM_response": final_answer,
                        "metadata": metadata if metadata is not None else ""
                    })

                    # Reset for the next pair.
                    model_params = None
                    user_raw_input = None
                    final_answer = None

        # After processing all events, check the results and write them to a new file.
        with open(new_file_path, 'w', encoding='utf-8') as outfile:
            if results:
                json.dump(results, outfile, indent=4)
                print(f"Results were written to {new_file_path}")
            else:
                print("No valid event pairs were found in the logs. No file was written.")

    except FileNotFoundError:
        print(f"The file {original_file_path} does not exist.")
    except json.JSONDecodeError:
        print("The file doesn't contain valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return new_file_path

# parse_old_json_format()
parse_2023_10_24_json_format()

