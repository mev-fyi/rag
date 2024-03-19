
from src.Llama_index_sandbox.data_ingestion_youtube.load.download_mp3 import main as download_mp3_main
from src.Llama_index_sandbox.data_ingestion_youtube.load.save_speaker_raw_diarized_audio_files import main as save_speaker_main
from src.Llama_index_sandbox.ingestion_pipeline import main as ingestion_pipeline_main


def run_all():
    download_mp3_main()
    save_speaker_main()
    ingestion_pipeline_main()


if __name__ == "__main__":
    run_all()
