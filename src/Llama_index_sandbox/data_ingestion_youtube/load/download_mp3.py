import asyncio
import itertools
import os
import argparse
from typing import List, Optional, Dict
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from googleapiclient.discovery import build
from dotenv import load_dotenv
# To download videos and transcripts from private Channels or Playlists
from google.oauth2.credentials import Credentials
import pandas as pd
import yt_dlp as ydlp

from src.Llama_index_sandbox import root_directory, YOUTUBE_VIDEO_DIRECTORY
from src.Llama_index_sandbox.utils import background, authenticate_service_account

# Load environment variables from the .env file
load_dotenv()
DOWNLOAD_AUDIO = os.environ.get('DOWNLOAD_AUDIO', 'True').lower() == 'true'
DOWNLOAD_TRANSCRIPTS = False


def download_audio_ydlp(video_url: str, output_path: str, audio_title: str):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_path}/{audio_title}.%(ext)s',
    }

    with ydlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


def get_video_info(credentials: Credentials, api_key: str, channel_id: str, max_results: int = 500000) -> List[dict]:
    """
    Retrieves video information (URL, ID, and title) from a YouTube channel using the YouTube Data API.

    Args:
        api_key (str): Your YouTube Data API key.
        channel_id (str): The YouTube channel ID.
        max_results (int, optional): Maximum number of results to retrieve. Defaults to 50.

    Returns:
        list: A list of dictionaries containing video URL, ID, and title from the channel.
    """
    # Initialize the YouTube API client
    if credentials is None:
        youtube = build('youtube', 'v3', developerKey=api_key)
    else:
        youtube = build('youtube', 'v3', credentials=credentials, developerKey=api_key)

    # Get the "Uploads" playlist ID
    channel_request = youtube.channels().list(
        part="contentDetails",
        id=channel_id,
        fields="items/contentDetails/relatedPlaylists/uploads"
    )
    channel_response = channel_request.execute()
    uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    # Fetch videos from the "Uploads" playlist
    video_info = []
    next_page_token = None

    while True:
        playlist_request = youtube.playlistItems().list(
            part="snippet",
            playlistId=uploads_playlist_id,
            maxResults=max_results,
            pageToken=next_page_token,
            fields="nextPageToken,items(snippet(publishedAt,resourceId(videoId),title))"
        )
        try:
            playlist_response = playlist_request.execute()
        except Exception as e:
            print(f"Error fetching videos for channel {channel_id}: {e}")
            return video_info
        items = playlist_response.get('items', [])

        for item in items:
            video_id = item["snippet"]["resourceId"]["videoId"]
            video_info.append({
                'url': f'https://www.youtube.com/watch?v={video_id}',
                'id': video_id,
                'title': item["snippet"]["title"],
                'publishedAt': item["snippet"]["publishedAt"]
            })

        next_page_token = playlist_response.get("nextPageToken")

        if next_page_token is None or len(video_info) >= max_results:
            break
    return video_info


@background
def parse_video(video_info: Dict[str, str], dir_path: str, youtube_videos_df) -> None:
    """
    Fetch and save the transcript of a YouTube video as a .txt file.

    Args:
        video_info (Dict[str, str]): A dictionary containing video information such as 'id', 'title', and 'publishedAt'.
        dir_path (str): The directory path where the transcript file will be saved.

    Returns:
        None
    """

    # Get video ID
    video_id = video_info['id']

    # Format video title and published date for file naming
    video_title = video_info['title'].replace('/', '_')

    # Define the paths of the files to check
    mp3_file_path = os.path.join(dir_path, f"{video_title}.mp3")
    json_file_path = os.path.join(dir_path, f"{video_title}_diarized_content.json")
    txt_file_path = os.path.join(dir_path, f"{video_title}_diarized_content_processed_diarzed.txt")

    # Check if any of the files already exist. If they do, return immediately.
    if os.path.exists(mp3_file_path) or os.path.exists(json_file_path) or os.path.exists(txt_file_path):
        print(f"Files for '{video_title}' already exist. Skipping download.")
        return

    # Similarly, replace sequences of spaces in the DataFrame's 'title' column
    youtube_videos_df['title'] = youtube_videos_df['title'].str.replace(' +', ' ', regex=True)

    # Now look for a match
    video_row = youtube_videos_df[youtube_videos_df['title'] == video_title]

    if video_row.empty:
        print(f"Video '{video_title}' not in shortlist of youtube videos. Skipping download.")
        # if the video title is not already in our list of videos, then do not download
        return

    strlen = len("yyyy-mm-dd")
    published_at = video_info['publishedAt'].replace(':', '-').replace('.', '-')[:strlen]
    video_title = f"{published_at}_{video_title}"

    # Create a specific directory for this video if it doesn't exist
    video_dir_path = os.path.join(dir_path, video_title)
    if not os.path.exists(video_dir_path):
        os.makedirs(video_dir_path)

    # Create the file paths for the transcript and the audio
    file_path = os.path.join(video_dir_path, f'{video_title}.txt')
    audio_file_path = os.path.join(video_dir_path, f'{video_title}.mp3')

    # Check if transcript exists, if not, download it
    if DOWNLOAD_TRANSCRIPTS and not os.path.exists(file_path):
        try:
            print(f"TRYING VIDEO: [{file_path}]")
            # Get the transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)

            # Write the transcript to a .txt file
            with open(file_path, 'w') as f:
                for line in transcript:
                    f.write(f"{line['text']} ")

            print(f'Successfully saved transcript for {video_info["url"]} as {file_path}')

        except TranscriptsDisabled:
            print(f"No transcripts available for {video_info['url']} with title {video_title}")

        except NoTranscriptFound:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
            except Exception as e:
                print(f"Error fetching translated transcript for {video_info['url']} with title {video_title}: {e}")
                return
        except Exception as e:
            print(f"Unknown error [{e}] for {video_title}")
    else:
        pass
        # print(f"Transcript already exists for {video_info['url']} with title {video_title}")
    # Check if audio exists, if not, download it
    if DOWNLOAD_AUDIO and not os.path.exists(audio_file_path):
        try:
            print(f"Downloading audio for {video_info['url']} to {video_dir_path}")
            download_audio_ydlp(video_info['url'], video_dir_path, video_title)
            print(f'Finished processing {video_info["url"]} with title {video_title}')

        except Exception as e:
            print(f'Error fetching .mp3 for {video_info["url"]} with title {video_title} and error: [{e}]')


def get_channel_id(credentials: Optional[Credentials], api_key: str, channel_name: str) -> Optional[str]:
    """
    Get the channel ID of a YouTube channel by its name.

    Args:
        api_key (str): Your YouTube Data API key.
        channel_name (str): The name of the YouTube channel.

    Returns:
        Optional[str]: The channel ID if found, otherwise None.
    """
    # Initialize the YouTube API client
    if credentials is None:
        youtube = build('youtube', 'v3', developerKey=api_key)
    else:
        youtube = build('youtube', 'v3', credentials=credentials, developerKey=api_key)

    # Create a search request to find the channel by name
    request = youtube.search().list(
        part='snippet',
        type='channel',
        q=channel_name,
        maxResults=1,
        fields='items(id(channelId))'
    )

    # Execute the request and get the response
    response = request.execute()

    # Get the list of items (channels) from the response
    items = response.get('items', [])

    # If there is at least one item, return the channel ID, otherwise return None
    if items:
        return items[0]['id']['channelId']
    else:
        return None


def get_playlist_title(credentials: Credentials, api_key: str, playlist_id: str) -> Optional[str]:
    """
    Retrieves the title of a YouTube playlist using the YouTube Data API.

    Args:
        api_key (str): Your YouTube Data API key.
        playlist_id (str): The YouTube playlist ID.

    Returns:
        Optional[str]: The title of the playlist if found, otherwise None.
    """
    # Initialize the YouTube API client
    if credentials is None:
        youtube = build('youtube', 'v3', developerKey=api_key)
    else:
        youtube = build('youtube', 'v3', credentials=credentials, developerKey=api_key)

    request = youtube.playlists().list(
        part='snippet',
        id=playlist_id,
        fields='items(snippet/title)',
        maxResults=1
    )
    response = request.execute()
    items = response.get('items', [])

    if items:
        return items[0]['snippet']['title']
    else:
        return None


def get_videos_from_playlist(credentials: Credentials, api_key: str, playlist_id: str, max_results: int = 5000) -> List[dict]:
    # Initialize the YouTube API client
    if credentials is None:
        youtube = build('youtube', 'v3', developerKey=api_key)
    else:
        youtube = build('youtube', 'v3', credentials=credentials, developerKey=api_key)

    video_info = []
    next_page_token = None

    while True:
        playlist_request = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=max_results,
            pageToken=next_page_token,
            fields="nextPageToken,items(snippet(publishedAt,resourceId(videoId),title))"
        )
        playlist_response = playlist_request.execute()
        items = playlist_response.get('items', [])

        for item in items:
            video_id = item["snippet"]["resourceId"]["videoId"]
            video_info.append({
                'url': f'https://www.youtube.com/watch?v={video_id}',
                'id': video_id,
                'title': item["snippet"]["title"],
                'publishedAt': item["snippet"]["publishedAt"]
            })

        next_page_token = playlist_response.get("nextPageToken")

        if next_page_token is None or len(video_info) >= max_results:
            break

    return video_info


def run(api_key: str, yt_channels: Optional[List[str]] = None, yt_playlists: Optional[List[str]] = None):
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

    dir_path = YOUTUBE_VIDEO_DIRECTORY

    # Iterate through the dictionary of channel IDs and channel names
    for channel_id, channel_name in yt_id_name.items():

        # Get video information from the channel
        video_info_list = get_video_info(credentials, api_key, channel_id)

        # Create a 'data' directory if it does not exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Create a subdirectory for the current channel if it does not exist
        dir_path += f'/{channel_name}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Iterate through video information, fetch transcripts, and save them as .txt files
        loop = asyncio.get_event_loop()
        args = [(video_info, dir_path, youtube_videos_df) for video_info in video_info_list]

        tasks = itertools.starmap(parse_video, args)
        loop.run_until_complete(asyncio.gather(*tasks))

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

            loop = asyncio.get_event_loop()
            args = [(video_info, dir_path, youtube_videos_df) for video_info in video_info_list]

            tasks = itertools.starmap(parse_video, args)
            loop.run_until_complete(asyncio.gather(*tasks))


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

    run(api_key, yt_channels, yt_playlists)

