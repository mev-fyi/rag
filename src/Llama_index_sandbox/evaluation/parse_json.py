import json
import os

from src.Llama_index_sandbox import root_directory


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
    new_file_path = os.path.join(subdirectory_path, 'parsed_2023-10-20_14:32:07.log.json')

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


parse_old_json_format()
