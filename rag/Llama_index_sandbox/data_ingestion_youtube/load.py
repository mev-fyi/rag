import assemblyai as aai


def transcribe_single_mp3():
    # URL of the file to transcribe
    FILE_URL = "/home/user/PycharmProjects/mev.fyi/data/youtube_video_transcripts/@bellcurvepodcast/2023-04-18_MEV in a Modular World | Jon Charbonneau, Robert Miller/2023-04-18_MEV in a Modular World | Jon Charbonneau, Robert Miller.mp4"

    config = aai.TranscriptionConfig(speaker_labels=True)

    transcriber = aai.Transcriber()

    # Start timing the transcription process
    start_time_transcription = time.time()

    transcript = transcriber.transcribe(
      FILE_URL,
      config=config
    )

    # End timing the transcription process
    end_time_transcription = time.time()
    transcription_duration = end_time_transcription - start_time_transcription

    # Path where the transcript file will be saved
    transcript_file_path = "/home/user/PycharmProjects/mev.fyi/data/youtube_video_transcripts/2023-04-18_MEV in a Modular World | Jon Charbonneau, Robert Miller/2023-04-18_MEV in a Modular World | Jon Charbonneau, Robert Miller_transcript.txt"


    # Create the directory if it does not exist
    transcript_directory = os.path.dirname(transcript_file_path)
    os.makedirs(transcript_directory, exist_ok=True)

    # Function to convert milliseconds to hours:minutes:seconds.milliseconds format
    def format_time(ms):
        seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


    # Opening the file in write mode to save the transcript
    with open(transcript_file_path, 'w') as file:
        for utterance in transcript.utterances:
            start_time = format_time(utterance.start)
            end_time = format_time(utterance.end)
            line_to_write = f"Speaker {utterance.speaker} ({start_time} to {end_time}): {utterance.text}\n"
            file.write(line_to_write)
            print(line_to_write)

    print(f"Transcript has been saved to {transcript_file_path}")
    print(f"The transcription process took {transcription_duration:.2f} seconds.")