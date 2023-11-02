import time
import logging
import os
from datetime import datetime
from functools import wraps

from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials


def root_directory() -> str:
    """
    Determine the root directory of the project based on the presence of '.git' directory.

    Returns:
    - str: The path to the root directory of the project.
    """
    current_dir = os.getcwd()

    while True:
        if '.git' in os.listdir(current_dir):
            return current_dir
        else:
            # Go up one level
            current_dir = os.path.dirname(current_dir)


def start_logging(log_prefix):
    # Ensure that root_directory() is defined and returns the path to the root directory

    # Create a 'logs' directory if it does not exist
    if not os.path.exists(f'{root_directory()}/logs/txt'):
        os.makedirs(f'{root_directory()}/logs/txt')

    # Get the current date and time
    now = datetime.now()
    timestamp_str = now.strftime('%Y-%m-%d_%H-%M')

    # Set up the logging level
    root_logger = logging.getLogger()

    # If handlers are already present, we can disable them.
    if root_logger.hasHandlers():
        # Clear existing handlers from the root logger
        root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    # Add handler to log messages to a file
    log_filename = f'{root_directory()}/logs/txt/{timestamp_str}_{log_prefix}.log'
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

    # Add handler to log messages to the standard output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)

    # Now, any logging.info() call will append the log message to the specified file and the standard output.
    logging.info(f'********* {log_prefix} LOGGING STARTED *********')



def timeit(func):
    """
    A decorator that logs the time a function takes to execute.

    Args:
        func (callable): The function being decorated.

    Returns:
        callable: The wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        The wrapper function to execute the decorated function and log its execution time.

        Args:
            *args: Variable length argument list to pass to the decorated function.
            **kwargs: Arbitrary keyword arguments to pass to the decorated function.

        Returns:
            The value returned by the decorated function.
        """
        logging.info(f"{func.__name__} STARTED.")
        start_time = time.time()

        # Call the decorated function and store its result.
        # *args and **kwargs are used to pass the arguments received by the wrapper
        # to the decorated function.
        result = func(*args, **kwargs)

        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        logging.info(f"{func.__name__} COMPLETED, took {int(minutes)} minutes and {seconds:.2f} seconds to run.\n")

        return result  # Return the result of the decorated function

    return wrapper


def authenticate_service_account(service_account_file: str) -> Credentials:
    """Authenticates using service account and returns the session."""

    credentials = ServiceAccountCredentials.from_service_account_file(
        service_account_file,
        scopes=["https://www.googleapis.com/auth/youtube.readonly"]
    )
    return credentials


def get_last_index_embedding_params():
    index_dir = f"{root_directory()}/.storage/research_pdf/"
    index = sorted(os.listdir(index_dir))[-1].split('_')
    index_date = index[0]
    embedding_model_name = index[1]
    embedding_model_chunk_size = int(index[2])
    chunk_overlap = int(index[3])
    vector_space_distance_metric = ''  # TODO 2023-11-02: save vector_space_distance_metric in index name
    return embedding_model_name, embedding_model_chunk_size, chunk_overlap, vector_space_distance_metric


import os
import fnmatch
import re

def find_matching_files(directory: str):
    mp3_files = []
    json_txt_files = []

    # 1. Recursively walk through the directory and collect paths to all .mp3, .json, and .txt files
    for dirpath, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, "*.mp3"):
            mp3_files.append(os.path.join(dirpath, filename))
        for filename in fnmatch.filter(filenames, "*.json"):
            json_txt_files.append(os.path.join(dirpath, filename))
        for filename in fnmatch.filter(filenames, "*.txt"):
            json_txt_files.append(os.path.join(dirpath, filename))

    matched_tuples = []

    for mp3_file in mp3_files:
        mp3_basename = os.path.basename(mp3_file).rsplit('.', 1)[0]
        for jt_file in json_txt_files:
            jt_basename = os.path.basename(jt_file).rsplit('.', 1)[0]

            # Remove prefix date if it exists
            jt_basename = re.sub(r'^\d{4}-\d{2}-\d{2}_', '', jt_basename)

            # Remove various suffixes
            jt_basename = re.sub(r'(_diarized_content(_processed_diarized)?)$', '', jt_basename)

            if mp3_basename == jt_basename:
                matched_tuples.append((mp3_file, jt_file))

    # 3. For each match, print the tuple and then later delete the .mp3 file
    for mp3_file, jt_file in matched_tuples:
        print((mp3_file, jt_file))
        if os.path.exists(mp3_file):
            os.remove(mp3_file)
            print(f"Deleting {mp3_file}")


import os
import pandas as pd
import shutil


def find_closest_match(video_title, df_titles):
    max_overlap = 0
    best_match = None
    for title in df_titles:
        overlap = sum(1 for a, b in zip(video_title, title) if a == b)
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = title
    return best_match


def move_remaining_mp3_to_their_subdirs():
    # Load the DataFrame
    videos_path = f"{root_directory()}/datasets/evaluation_data/youtube_videos.csv"
    youtube_videos_df = pd.read_csv(videos_path)
    youtube_videos_df['title'] = youtube_videos_df['title'].str.replace(' +', ' ', regex=True)
    youtube_videos_df['title'] = youtube_videos_df['title'].str.replace('"', '', regex=True)

    # Get a list of all mp3 files in the directory and subdirectories
    mp3_files = []
    for subdir, dirs, files in os.walk(f"{root_directory()}/datasets/evaluation_data/diarized_youtube_content_2023-10-06"):
        for file in files:
            if file.endswith(".mp3"):
                mp3_files.append(os.path.join(subdir, file))

    df_titles = youtube_videos_df['title'].tolist()
    # Process each mp3 file
    for mp3_file in mp3_files:
        # Extract the segment after the last "/"
        video_title = mp3_file.split('/')[-1].rsplit('.', 1)[0]
        # Replace double spaces with a single space
        video_title = video_title.replace('  ', ' ').strip()

        # Check if mp3 file is already in a directory matching its name
        containing_dir = os.path.basename(os.path.dirname(mp3_file))
        if video_title == containing_dir:
            continue

        # video_row = youtube_videos_df[youtube_videos_df['title'].str.contains(video_title, case=False, na=False, regex=False)]
        best_match = find_closest_match(video_title, df_titles)
        video_row = youtube_videos_df[youtube_videos_df['title'] == best_match]

        if not video_row.empty:
            published_date = video_row.iloc[0]['published_date']
            new_dir_name = f"{published_date}_{video_title}"
            new_dir_path = os.path.join(os.path.dirname(mp3_file), new_dir_name)
            os.makedirs(new_dir_path, exist_ok=True)
            new_file_name = f"{published_date}_{video_title}.mp3"
            new_file_path = os.path.join(new_dir_path, new_file_name)
            print(f"Moved video {best_match} to {new_file_path}!")
            shutil.move(mp3_file, new_file_path)
        else:
            print(f"No matching video title found in DataFrame for: {video_title}")


def merge_directories(base_path):
    '''
    This function walks through all subdirectories and merges the contents of directories that have
    names differing only by the pipe character used, from fullwidth to ASCII. Files from the fullwidth
    pipe directory are moved to the ASCII pipe directory, and if a file with the same name exists, the
    file from the fullwidth pipe directory is deleted. After the merge, the fullwidth pipe directory is
    deleted if empty.

    Args:
        base_path: The base directory path to start searching from.

    Returns: None
    '''

    # Helper function to rename the pipe character
    def standardize_name(dir_or_file_name):
        return dir_or_file_name.replace('｜', '|')

    # Track directories to be removed after processing
    dirs_to_remove = []

    # Walk through the directory structure
    for root, dirs, _ in os.walk(base_path):
        # Map of standard directory names to their full paths
        standard_dirs = {}

        # First pass to fill in the mapping
        for dir_name in dirs:
            standard_dirs[standardize_name(dir_name)] = os.path.join(root, dir_name)

        # Second pass to perform the merging
        for dir_name in dirs:
            standard_name = standardize_name(dir_name)
            src = os.path.join(root, dir_name)
            dst = standard_dirs[standard_name]

            # Only proceed if the directory names actually differ (by the pipe character)
            if src != dst:
                if not os.path.exists(dst):
                    # If the destination doesn't exist, simply rename the directory
                    os.rename(src, dst)
                    print(f"Renamed {src} to {dst}")
                else:
                    # Merge contents
                    for item in os.listdir(src):
                        src_item = os.path.join(src, item)
                        dst_item = os.path.join(dst, standardize_name(item))
                        if os.path.exists(dst_item):
                            # If there is a conflict, delete the source item
                            os.remove(src_item)
                            print(f"Deleted due to conflict: {src_item}")
                        else:
                            shutil.move(src_item, dst_item)
                            print(f"Moved {src_item} to {dst_item}")

                    # Add to list of directories to remove if they are empty
                    dirs_to_remove.append(src)

    # Remove the source directories if they are empty
    for dir_to_remove in dirs_to_remove:
        if not os.listdir(dir_to_remove):
            os.rmdir(dir_to_remove)
            print(f"Removed empty directory: {dir_to_remove}")
        else:
            print(f"Directory {dir_to_remove} is not empty after merge. Please check contents.")

# Usage


if __name__ == '__main__':
    directory = f"{root_directory()}/datasets/evaluation_data/diarized_youtube_content_2023-10-06"
    # find_matching_files(directory)
    # move_remaining_mp3_to_their_subdirs()
    merge_directories(f"{root_directory()}/datasets/evaluation_data/diarized_youtube_content_2023-10-06")
