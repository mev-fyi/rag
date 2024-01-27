import json
import logging

import assemblyai as aai
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import re

from src.Llama_index_sandbox import YOUTUBE_VIDEO_DIRECTORY

load_dotenv()
api_key = os.environ.get('ASSEMBLY_AI_API_KEY')

if not api_key:
    raise EnvironmentError("ASSEMBLY_AI_API_KEY environment variable not found. Please set it before running the script.")

aai.settings.api_key = api_key


def is_valid_filename(filename):
    """Check if the filename starts with 'yyyy-mm-dd_' format."""
    return re.match(r'^\d{4}-\d{2}-\d{2}_', filename)


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
    try:
        transcript_file_path = os.path.splitext(file_path)[0] + "_diarized_content.json"

        if os.path.exists(transcript_file_path):
            logging.info(f"Content for {file_path.split('/')[-1].replace('.mp3', '.json')} already diarized. Skipping.")
            return

        if not os.path.exists(file_path):
            logging.warning(f"File {file_path} not found.")
            return

        logging.info(f"Diarization started for {file_path}")

        config = aai.TranscriptionConfig(speaker_labels=True)
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_path, config=config)

        utterances_dicts = [utterance_to_dict(utterance) for utterance in transcript.utterances]

        with open(transcript_file_path, 'w') as file:
            json.dump(utterances_dicts, file, indent=4)

        logging.info(f"Transcript for {file_path} saved to {transcript_file_path}")

    except Exception as e:
        logging.error(f"Error transcribing {file_path}: {e}")


def main():
    """Main function to transcribe files."""
    try:
        data_path = YOUTUBE_VIDEO_DIRECTORY
        mp3_files = []

        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".mp3") and is_valid_filename(file):
                    mp3_files.append(os.path.join(root, file))

        if not mp3_files:
            logging.warning("No MP3 files found to transcribe.")
            return

        max_workers = os.cpu_count()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(transcribe_and_save, mp3_files)

    except Exception as e:
        logging.error(f"An error occurred in main: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
