import json
import assemblyai as aai
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from src.Llama_index_sandbox import YOUTUBE_VIDEO_DIRECTORY

load_dotenv()
aai.settings.api_key = os.environ.get('ASSEMBLY_AI_API_KEY')


def root_directory() -> str:
    current_dir = os.getcwd()

    while True:
        if '.git' in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)


def utterance_to_dict(utterance) -> dict:
    """
    Convert an Utterance object to a dictionary.
    """
    return {
        'text': utterance.text,
        'start': utterance.start,
        'end': utterance.end,
        'confidence': utterance.confidence,
        'channel': utterance.channel,
        'speaker': utterance.speaker,
        'words': [{
            'text': word.text,
            'start': word.start,
            'end': word.end,
            'confidence': word.confidence,
            'channel': word.channel,
            'speaker': word.speaker
        } for word in utterance.words]
    }


def transcribe_and_save(file_path):
    # Name the transcript file by appending "_diarized_content.json" to the MP3 filename
    transcript_file_path = os.path.splitext(file_path)[0] + "_diarized_content.json"

    # Check if the content has been diarized already
    if os.path.exists(transcript_file_path):
        print(f"Content for {file_path.split('/')[-1].replace('.mp3', '.json')} has already been diarized. Skipping...\n")
        return

    if not os.path.exists(file_path):
        # print(f"File {file_path} not found. Skipping...")
        return
    print(f"Diarization started for {file_path}")

    config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path, config=config)

    # Convert utterances to dictionaries
    utterances_dicts = [utterance_to_dict(utterance) for utterance in transcript.utterances]

    # Save the utterances to a JSON file
    with open(transcript_file_path, 'w') as file:
        json.dump(utterances_dicts, file, indent=4)

    filename = file_path.split('/')[-1]
    print(f"\nTranscript for {filename} has been saved to {transcript_file_path}")


def main():
    data_path = YOUTUBE_VIDEO_DIRECTORY
    mp3_files = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".mp3"):
                mp3_files.append(os.path.join(root, file))

    max_workers = os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(transcribe_and_save, mp3_files)


if __name__ == "__main__":
    main()
