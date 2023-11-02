import asyncio
import glob
import itertools
import os
import argparse
from typing import List, Optional, Dict
from dotenv import load_dotenv
# To download videos and transcripts from private Channels or Playlists
import pandas as pd
import yt_dlp as ydlp
from yt_dlp import DownloadError

from src.Llama_index_sandbox import root_directory, YOUTUBE_VIDEO_DIRECTORY
from src.Llama_index_sandbox.data_ingestion_youtube.load.utils import get_videos_from_playlist, get_channel_id, get_playlist_title, get_video_info
from src.Llama_index_sandbox.utils import authenticate_service_account, move_remaining_mp3_to_their_subdirs, clean_fullwidth_characters, merge_directories, delete_mp3_if_text_or_json_exists

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=os.cpu_count())

# Load environment variables from the .env file
load_dotenv()
DOWNLOAD_AUDIO = os.environ.get('DOWNLOAD_AUDIO', 'True').lower() == 'true'


def chunked_iterable(iterable, size):
    """Splits an iterable into chunks of a specified size."""
    iterator = iter(iterable)
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, size - 1))


async def download_audio_batch(video_infos, ydl_opts):
    """
    This function downloads multiple audio files based on the provided list of video information.
    """
    urls = [info['url'] for info in video_infos]  # Extract URLs from the video information
    print(f"Dowloading batch of urls {urls}")
    with ydlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download(urls)  # Download all videos using a single command
        except DownloadError:
            print(f"Download error for {urls}")
        except Exception as e:
            print(f"Other error not caught by handler: {e}")
            pass


async def video_valid_for_processing(video_title, youtube_videos_df, dir_path):
    # This helper function checks if the video_title exists in any of the filenames in the specified directory
    video_title = video_title.replace('/', '_')

    def title_exists_in_files(directory, suffix):
        for filename in os.listdir(directory):
            if video_title in filename and filename.endswith(suffix):
                return True
        return False

    # Recursively check in dir_path and its subdirectories
    for root, dirs, files in os.walk(dir_path):
        if (title_exists_in_files(root, ".mp3") or
                title_exists_in_files(root, "_diarized_content.json") or
                title_exists_in_files(root, "_diarized_content_processed_diarized.txt") or
                title_exists_in_files(root, "_content_processed_diarized.txt")):
            return False

    # The same code for dataframe handling
    youtube_videos_df['title'] = youtube_videos_df['title'].str.replace(' +', ' ', regex=True)
    youtube_videos_df['title'] = youtube_videos_df['title'].str.replace('"', '', regex=True)

    video_row = youtube_videos_df[youtube_videos_df['title'] == video_title]

    if video_row.empty:
        print(f"SKIPPING '{video_title}'")
        return False
    else:
        print(f"ADDING '{video_title}'")
    return True


async def prepare_download_info(video_info, dir_path, video_title):
    strlen = len("yyyy-mm-dd")
    published_at = video_info['publishedAt'].replace(':', '-').replace('.', '-')[:strlen]
    video_title = f"{published_at}_{video_title}"

    # Create a specific directory for this video if it doesn't exist
    video_dir_path = os.path.join(dir_path, video_title)
    if not os.path.exists(video_dir_path):
        os.makedirs(video_dir_path)

    audio_file_path = os.path.join(video_dir_path, f'{video_title}.mp3')
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{video_dir_path}/{video_title}.%(ext)s',
    }
    return ydl_opts, audio_file_path


async def process_video_batches(video_info_list, dir_path, youtube_videos_df, batch_size=5):
    video_batches = list(chunked_iterable(video_info_list, batch_size))
    # TODO 2023-10-31: fix somehow the addition of spaces before colons : e.g. for DVT and for Defi panel
    #  The different pipes somehow. Check the match method in utils.py
    # Create a common download option for all videos in the batch.
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        # NOTE 2023-10-31: we do not pass the video title and its subdirectory because we process several URL at once per batch.
        # Therefore we cannot match tuples of ydl_opts and their URLs.
        # So we'll have discerpeancies.
        'outtmpl': f'{dir_path}/%(title)s.%(ext)s',
    }

    tasks = []
    for batch_info in video_batches:
        valid_videos = [video for video in batch_info if await video_valid_for_processing(video['title'], youtube_videos_df, dir_path)]
        if valid_videos:
            # We're creating a task for each batch of valid videos.
            task = asyncio.create_task(download_audio_batch(valid_videos, ydl_opts))
            tasks.append(task)

    # Now we run the download tasks concurrently.
    await asyncio.gather(*tasks)


async def run(api_key: str, yt_channels: Optional[List[str]] = None, yt_playlists: Optional[List[str]] = None):
    """
    Run function that takes a YouTube Data API key and a list of YouTube channel names, fetches video transcripts,
    and saves them as .txt files in a data directory.

    Args:
        yt_playlists:
        api_key (str): Your YouTube Data API key.
        yt_channels (List[str]): A list of YouTube channel names.
    """
    service_account_file = os.environ.get('SERVICE_ACCOUNT_FILE')
    credentials = None

    if service_account_file:
        credentials = authenticate_service_account(service_account_file)
        print("Service account file found. Proceeding with public channels, playlists, or private videos if accessible via Google Service Account.")
    else:
        print("No service account file found. Proceeding with public channels or playlists.")

    # Create a dictionary with channel IDs as keys and channel names as values
    yt_id_name = {get_channel_id(credentials=credentials, api_key=api_key, channel_name=name): name for name in yt_channels}

    videos_path = f"{root_directory()}/datasets/evaluation_data/youtube_videos.csv"
    youtube_videos_df = pd.read_csv(videos_path)

    # Iterate through the dictionary of channel IDs and channel names
    for channel_id, channel_name in yt_id_name.items():

        dir_path = YOUTUBE_VIDEO_DIRECTORY
        # Get video information from the channel
        video_info_list = get_video_info(credentials, api_key, channel_id)

        # Create a 'data' directory if it does not exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Create a subdirectory for the current channel if it does not exist
        dir_path += f'{channel_name}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        await process_video_batches(video_info_list, dir_path, youtube_videos_df)

    if yt_playlists:
        for playlist_id in yt_playlists:
            playlist_title = get_playlist_title(credentials, api_key, playlist_id)
            # Ensure the title is filesystem-friendly (replacing slashes, for example)
            playlist_title = playlist_title.replace('/', '_') if playlist_title else f"playlist_{playlist_id}"

            video_info_list = get_videos_from_playlist(credentials, api_key, playlist_id)

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            dir_path += f'/{playlist_title}'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            await process_video_batches(video_info_list, dir_path, youtube_videos_df)

    # clean up because downloaded file names have full-width characters instead of ASCII
    directory = f"{root_directory()}/datasets/evaluation_data/diarized_youtube_content_2023-10-06"
    clean_fullwidth_characters(directory)
    move_remaining_mp3_to_their_subdirs()
    merge_directories(directory)
    delete_mp3_if_text_or_json_exists(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch YouTube video transcripts.')
    parser.add_argument('--api_key', type=str, help='YouTube Data API key')
    parser.add_argument('--channels', nargs='+', type=str, help='YouTube channel names or IDs')
    parser.add_argument('--playlists', nargs='+', type=str, help='YouTube playlist IDs')

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('YOUTUBE_API_KEY')
    if not api_key:
        raise ValueError("No API key provided. Please provide an API key via command line argument or .env file.")

    yt_channels = args.channels or os.environ.get('YOUTUBE_CHANNELS')
    if yt_channels:
        yt_channels = [channel.strip() for channel in yt_channels.split(',')]

    yt_playlists = args.playlists or os.environ.get('YOUTUBE_PLAYLISTS')
    if yt_playlists:
        yt_playlists = [playlist.strip() for playlist in yt_playlists.split(',')]

    if not yt_channels and not yt_playlists:
        raise ValueError(
            "No channels or playlists provided. Please provide channel names, IDs, or playlist IDs via command line argument or .env file.")

    asyncio.run(run(api_key, yt_channels, yt_playlists))

