import json
import os
import concurrent.futures

from rag.Llama_index_sandbox.utils import root_directory


def format_time(ms):
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def process_utterance(utterance, sentence_count):
    output = []
    current_speaker = utterance['speaker']
    current_start = utterance['start']
    current_content = []
    current_sentence_count = 0

    for word in utterance['words']:
        current_content.append(word['text'])

        if "." in word['text']:
            current_sentence_count += 1

        if current_sentence_count == sentence_count:
            formatted_start = format_time(current_start)
            formatted_end = format_time(word['end'])
            segment = f"{formatted_start} - {formatted_end}, Speaker {current_speaker}: {' '.join(current_content)}"
            output.append(segment)

            # Reset for next segment
            current_content = []
            current_sentence_count = 0
            current_start = word['end']

    if current_content:
        formatted_start = format_time(current_start)
        formatted_end = format_time(utterance['end'])
        segment = f"{formatted_start} - {formatted_end}, Speaker {current_speaker}: {' '.join(current_content)}"
        output.append(segment)

    return output


def process_transcript(file_path, sentence_count=3):
    print(f"Processing: {file_path.split('/')[-1]}")
    with open(file_path, 'r') as f:
        data = json.load(f)

    all_segments = []
    for utterance in data:
        all_segments.extend(process_utterance(utterance, sentence_count))

    # Save the results locally
    output_filename = os.path.splitext(os.path.basename(file_path))[0] + "_processed_diarized.txt"
    output_path = os.path.join(os.path.dirname(file_path), output_filename)
    with open(output_path, 'w') as output_file:
        for segment in all_segments:
            output_file.write(segment + '\n')
    print(f"Saved {output_filename}")


if __name__ == "__main__":
    data_directory = f"{root_directory()}/datasets/evaluation_data/diarized_youtube_content_2023-10-04/"

    files_to_process = []
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file.endswith("_diarized_content.json"):
                files_to_process.append(os.path.join(root, file))

    # Process the files in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_transcript, files_to_process)

