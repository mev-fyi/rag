import time
import logging
import os
import inspect
# import fitz
from datetime import datetime
from functools import wraps
import shutil

from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials

import subprocess

from llama_index.llms import ChatMessage, MessageRole


import os
import subprocess

def root_directory() -> str:
    """
    Determine the root directory of the project. It checks if it's running in a Docker container and adjusts accordingly.

    Returns:
    - str: The path to the root directory of the project.
    """

    # Check if running in a Docker container
    if os.path.exists('/.dockerenv'):
        # If inside a Docker container, use '/app' as the root directory
        return '/app'

    # If not in a Docker container, try to use the git command to find the root directory
    try:
        git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.STDOUT)
        return git_root.strip().decode('utf-8')
    except subprocess.CalledProcessError:
        # Git command failed, which might mean we're not in a Git repository
        # Fall back to manual traversal
        pass
    except Exception as e:
        # Some other error occurred while trying to execute git command
        print(f"An error occurred while trying to find the Git repository root: {e}")

    # Manual traversal if git command fails
    current_dir = os.getcwd()
    root = os.path.abspath(os.sep)
    traversal_count = 0  # Track the number of levels traversed

    while current_dir != root:
        try:
            if 'src' in os.listdir(current_dir):
                print(f"Found root directory: {current_dir}")
                return current_dir
            current_dir = os.path.dirname(current_dir)
            traversal_count += 1
            print(f"Traversal count # {traversal_count}")
            if traversal_count > 10:
                raise Exception("Exceeded maximum traversal depth (more than 10 levels).")
        except PermissionError as e:
            # Could not access a directory due to permission issues
            raise Exception(f"Permission denied when accessing directory: {current_dir}") from e
        except FileNotFoundError as e:
            # The directory was not found, which should not happen unless the filesystem is changing
            raise Exception(f"The directory was not found: {current_dir}") from e
        except OSError as e:
            # Handle any other OS-related errors
            raise Exception("An OS error occurred while searching for the Git repository root.") from e

    # If we've reached this point, it means we've hit the root of the file system without finding a .git directory
    raise Exception("Could not find the root directory of the project. Please make sure you are running this script from within a Git repository.")


def start_logging(log_prefix):
    # Ensure that root_directory() is defined and returns the path to the root directory

    logs_dir = f'{root_directory()}/logs/txt'

    # Create a 'logs' directory if it does not exist, with exist_ok=True to avoid FileExistsError
    os.makedirs(logs_dir, exist_ok=True)

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
    log_filename = f'{logs_dir}/{timestamp_str}_{log_prefix}.log'
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
    A decorator that logs the time a function takes to execute along with the directory and filename.

    Args:
        func (callable): The function being decorated.

    Returns:
        callable: The wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        The wrapper function to execute the decorated function and log its execution time and location.

        Args:
            *args: Variable length argument list to pass to the decorated function.
            **kwargs: Arbitrary keyword arguments to pass to the decorated function.

        Returns:
            The value returned by the decorated function.
        """
        if os.getenv('ENVIRONMENT') == 'LOCAL':
            # Get the current file's path and extract directory and filename
            file_path = inspect.getfile(func)
            directory, filename = os.path.split(file_path)
            dir_name = os.path.basename(directory)

            # Log start of function execution
            logging.info(f"{dir_name}.{filename}.{func.__name__} STARTED.")
            start_time = time.time()

            # Call the decorated function and store its result
            result = func(*args, **kwargs)

            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)

            # Log end of function execution
            logging.info(f"{dir_name}.{filename}.{func.__name__} COMPLETED, took {int(minutes)} minutes and {seconds:.2f} seconds to run.\n")

            return result
        else:
            # If not in 'LOCAL' environment, just call the function without timing
            return func(*args, **kwargs)

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
    vector_space_distance_metric = 'cosine'  # TODO 2023-11-02: save vector_space_distance_metric in index name
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


import pandas as pd


def find_closest_match(video_title, df_titles):
    max_overlap = 0
    best_match = None
    for title in df_titles:
        # Ensure title is a string before iterating
        title_str = str(title)
        overlap = sum(1 for a, b in zip(video_title, title_str) if a == b)
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = title_str
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


def move_remaining_txt_to_their_subdirs():
    # Load the DataFrame
    videos_path = f"{root_directory()}/datasets/evaluation_data/youtube_videos.csv"
    youtube_videos_df = pd.read_csv(videos_path)
    youtube_videos_df['title'] = youtube_videos_df['title'].str.replace(' +', ' ', regex=True)
    youtube_videos_df['title'] = youtube_videos_df['title'].str.replace('"', '', regex=True)

    # Get a list of all txt files in the directory and subdirectories
    txt_files = []
    for subdir, dirs, files in os.walk(f"{root_directory()}/datasets/evaluation_data/diarized_youtube_content_2023-10-06"):
        for file in files:
            if file.endswith("_diarized_content_processed_diarized.txt"):
                txt_files.append(os.path.join(subdir, file))

    df_titles = youtube_videos_df['title'].tolist()
    # Process each txt file
    for txt_file in txt_files:
        # Extract the segment after the last "/"
        extension = "_diarized_content_processed_diarized.txt"
        video_title = txt_file.replace(extension, '').split('/')[-1].rsplit('.', 1)[0]
        # Replace double spaces with a single space
        video_title = video_title.replace('  ', ' ').strip()

        # video_row = youtube_videos_df[youtube_videos_df['title'].str.contains(video_title, case=False, na=False, regex=False)]
        best_match = find_closest_match(video_title, df_titles)
        video_row = youtube_videos_df[youtube_videos_df['title'] == best_match]

        if not video_row.empty:
            published_date = video_row.iloc[0]['published_date']
            new_dir_name = f"{published_date}_{video_title}"

            # Check if txt file is already in a directory matching its name
            containing_dir = os.path.basename(os.path.dirname(txt_file))
            if new_dir_name == containing_dir:
                continue

            new_dir_path = os.path.join(os.path.dirname(txt_file), new_dir_name)
            os.makedirs(new_dir_path, exist_ok=True)
            new_file_name = f"{published_date}_{video_title}{extension}"
            new_file_path = os.path.join(new_dir_path, new_file_name)
            if os.path.exists(new_file_path):
                print(f"Deleted {txt_file} because {new_file_path} already exists")
                os.remove(txt_file)
            else:
                print(f"Moved video {txt_file} to {new_file_path}!")
                shutil.move(txt_file, new_file_path)
        else:
            print(f"No matching video title found in DataFrame for: {video_title}")


def move_remaining_json_to_their_subdirs():
    # Load the DataFrame
    videos_path = f"{root_directory()}/datasets/evaluation_data/youtube_videos.csv"
    youtube_videos_df = pd.read_csv(videos_path)
    youtube_videos_df['title'] = youtube_videos_df['title'].str.replace(' +', ' ', regex=True)
    youtube_videos_df['title'] = youtube_videos_df['title'].str.replace('"', '', regex=True)

    # Get a list of all json files in the directory and subdirectories
    json_files = []
    for subdir, dirs, files in os.walk(f"{root_directory()}/datasets/evaluation_data/diarized_youtube_content_2023-10-06"):
        for file in files:
            if file.endswith("_diarized_content.json"):
                json_files.append(os.path.join(subdir, file))

    df_titles = youtube_videos_df['title'].tolist()
    # Process each json file
    for json_file in json_files:
        # Extract the segment after the last "/"
        extension = "_diarized_content.json"
        video_title = json_file.replace(extension, '').split('/')[-1].rsplit('.', 1)[0]
        # Replace double spaces with a single space
        video_title = video_title.replace('  ', ' ').strip()

        # video_row = youtube_videos_df[youtube_videos_df['title'].str.contains(video_title, case=False, na=False, regex=False)]
        best_match = find_closest_match(video_title, df_titles)
        video_row = youtube_videos_df[youtube_videos_df['title'] == best_match]

        if not video_row.empty:
            published_date = video_row.iloc[0]['published_date']
            new_dir_name = f"{published_date}_{video_title}"

            # Check if json file is already in a directory matching its name
            containing_dir = os.path.basename(os.path.dirname(json_file))
            if new_dir_name == containing_dir:
                continue

            new_dir_path = os.path.join(os.path.dirname(json_file), new_dir_name)
            os.makedirs(new_dir_path, exist_ok=True)
            new_file_name = f"{published_date}_{video_title}{extension}"
            new_file_path = os.path.join(new_dir_path, new_file_name)
            if os.path.exists(new_file_path):
                print(f"Deleted {json_file} because {new_file_path} already exists")
                os.remove(json_file)
            else:
                print(f"Moved video {json_file} to {new_file_path}!")
                shutil.move(json_file, new_file_path)
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
        return dir_or_file_name.replace('：', ':')

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


def replace_fullwidth_colon_and_clean():
    base_path = f"{root_directory()}/datasets/evaluation_data/diarized_youtube_content_2023-10-06"

    for root, dirs, files in os.walk(base_path):
        json_files = set()

        # First, collect all .json filenames without extension
        for file in files:
            if file.endswith('.json'):
                json_files.add(file[:-5])  # Removes the '.json' part

        # Next, iterate over files and process them
        for file in files:
            original_file_path = os.path.join(root, file)
            if '：' in file:
                # Replace the fullwidth colon with a standard colon
                new_file_name = file.replace('｜', '|')   # return dir_or_file_name.replace('｜', '|')
                new_file_path = os.path.join(root, new_file_name)

                if os.path.exists(new_file_path):
                    # If the ASCII version exists, delete the fullwidth version
                    print(f"Deleted {original_file_path}")
                    os.remove(original_file_path)
                else:
                    # Otherwise, rename the file
                    print(f"Renamed {original_file_path} to {new_file_path}")
                    os.rename(original_file_path, new_file_path)

            # If a corresponding .json file exists, delete the .mp3 file
            if file[:-4] in json_files and file.endswith('.mp3'):
                os.remove(original_file_path)
                print(f"Deleted .mp3 file {original_file_path} because a corresponding .json exists")


def fullwidth_to_ascii(char):
    """Converts a full-width character to its ASCII equivalent."""
    # Full-width range: 0xFF01-0xFF5E
    # Corresponding ASCII range: 0x21-0x7E
    fullwidth_offset = 0xFF01 - 0x21
    return chr(ord(char) - fullwidth_offset) if 0xFF01 <= ord(char) <= 0xFF5E else char


def clean_fullwidth_characters(base_path):
    for root, dirs, files in os.walk(base_path, topdown=False):  # topdown=False to start from the innermost directories
        # First handle the files in the directories
        for file in files:
            new_file_name = ''.join(fullwidth_to_ascii(char) for char in file)
            original_file_path = os.path.join(root, file)
            new_file_path = os.path.join(root, new_file_name)

            if new_file_name != file:
                if os.path.exists(new_file_path):
                    # If the ASCII version exists, delete the full-width version
                    os.remove(original_file_path)
                    print(f"Deleted {original_file_path}")
                else:
                    # Otherwise, rename the file
                    os.rename(original_file_path, new_file_path)
                    print(f"Renamed {original_file_path} to {new_file_path}")

        # Then handle directories
        for dir in dirs:
            new_dir_name = ''.join(fullwidth_to_ascii(char) for char in dir)
            original_dir_path = os.path.join(root, dir)
            new_dir_path = os.path.join(root, new_dir_name)

            if new_dir_name != dir:
                if os.path.exists(new_dir_path):
                    # If the ASCII version exists, delete the full-width version and its contents
                    shutil.rmtree(original_dir_path)
                    print(f"Deleted directory and all contents: {original_dir_path}")
                else:
                    # Otherwise, rename the directory
                    os.rename(original_dir_path, new_dir_path)
                    print(f"Renamed {original_dir_path} to {new_dir_path}")


def delete_mp3_if_text_or_json_exists(base_path):
    for root, dirs, _ in os.walk(base_path):
        for dir in dirs:
            subdir_path = os.path.join(root, dir)
            # Get a list of files in the current subdirectory
            files = os.listdir(subdir_path)
            # Filter out .mp3, .txt and .json files
            mp3_files = [file for file in files if file.endswith('.mp3')]
            txt_json_files = [file for file in files if file.endswith('.txt') or file.endswith('.json')]

            if mp3_files:
                # If there are both .mp3 and (.txt or .json) files, delete the .mp3 files
                if txt_json_files:
                    for mp3_file in mp3_files:
                        mp3_file_path = os.path.join(subdir_path, mp3_file)
                        print(f"Deleted .mp3 file: {mp3_file_path}")
                        os.remove(mp3_file_path)
                else:
                    # If there are only .mp3 files, print their names and containing directory
                    for mp3_file in mp3_files:
                        pass
                        # print(f".mp3 file without .txt or .json: {mp3_file} in directory {subdir_path}")


def print_frontend_content():
    import os

    # Define the list of relative paths of the files you want to print
    file_paths = [
        # f"{root_directory()}/../rag_app_vercel/app/app/api/auth/[...nextauth]/route.ts",
        f"{root_directory()}/../rag_app_vercel/app/app/actions.ts",
        f"{root_directory()}/../rag_app_vercel/app/app/api/chat/route.ts",
        # f"{root_directory()}/../rag_app_vercel/app/chat/[id]/server-logic.ts",
        f"{root_directory()}/../rag_app_vercel/app/app/api/chat/[id]/page.tsx",
        # f"{root_directory()}/../rag_app_vercel/app/pages/chat.tsx",
        # f"{root_directory()}/../rag_app_vercel/app/pages/index.tsx",
        f"{root_directory()}/../rag_app_vercel/app/auth.ts",
        # f"{root_directory()}/../rag_app_vercel/app/components/chat.tsx",
        # f"{root_directory()}/../rag_app_vercel/app/components/chat-list.tsx",
        # f"{root_directory()}/../rag_app_vercel/app/components/chat-message.tsx",
        # f"{root_directory()}/../rag_app_vercel/app/components/chat-panel.tsx",
        # f"{root_directory()}/../rag_app_vercel/app/lib/hooks/use-chat-service.tsx",
    ]

    # file_path = 'app.py'
    # print("Here is the content of the app.py backend:")
    # with open(file_path, 'r') as file:
    #     content = file.read()
    #     print(f"{file_path}\n```\n{content}```\n")

    print("\n\nHere is the content of the frontend files:")
    # Iterate through the list, printing the content of each file
    for file_path in file_paths:
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                print(f"`{file_path.replace('/home/user/PycharmProjects/rag/../rag_app_vercel/','')}`\n```\n{content}\n```\n\n")
        else:
            print(f"{file_path}\n```File not found```")


import os
import zipfile

def save_data_into_zip ():
    def zip_files(directory, file_extension, zip_file):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(file_extension):
                    zip_file.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory))


    zip_filename = "collected_documents.zip"

    # Create a zip file
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        # Add all .pdf files from baseline_evaluation_research_papers_2023-10-05
        zip_files(f'{root_directory()}/datasets/evaluation_data/baseline_evaluation_research_papers_2023-10-05', '.pdf', zipf)

        # Add all .txt files from nested directories in diarized_youtube_content_2023-10-06
        zip_files(f'{root_directory()}/datasets/evaluation_data/diarized_youtube_content_2023-10-06', '.txt', zipf)

    print(f"Files zipped into {zip_filename}")


def copy_txt_files_to_transcripts(rootdir=root_directory()):
    source_dir = os.path.join(rootdir, 'datasets', 'evaluation_data', 'diarized_youtube_content_2023-10-06')
    target_dir = os.path.join(rootdir, 'datasets', 'evaluation_data', 'transcripts')

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Copy all .txt files from nested subdirectories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.txt'):
                source_file = os.path.join(root, file)
                shutil.copy(source_file, target_dir)

    print(f"All .txt files copied to {target_dir}")


def process_messages(data):
    try:
        messages = data["chat_history"]
    except KeyError:
        # Handle the absence of chat_history key more gracefully
        return None
    chat_messages = []

    for message in messages:
        # Create a ChatMessage object for each message
        chat_message = ChatMessage(
            role=MessageRole(message.get("role", "user").lower()),  # Convert the role to Enum
            content=message.get("content", ""),
            additional_kwargs=message.get("additional_kwargs", {})  # Assuming additional_kwargs is part of your message structure
        )
        chat_messages.append(chat_message)

    return chat_messages


def delete_redundant_directories(root_path):
    # Create a list to collect directories to be deleted
    directories_to_delete = []

    # Walk through the directory
    for subdir, dirs, files in os.walk(root_path, topdown=False):  # Note the 'topdown=False' parameter
        for dir in dirs:
            # Construct the path to the current directory
            current_dir_path = os.path.join(subdir, dir)
            # Check if directory name ends with the specified suffixes
            if dir.endswith('_diarized_content') or dir.endswith('_diarized_content_processed_diarized'):
                # Construct the file names that should exist in the parent directory
                json_file = dir.split('_', 1)[-1] + '_diarized_content.json'
                txt_file = dir.split('_', 1)[-1] + '_diarized_content_processed_diarized.txt'
                # Construct the paths to the files that should exist
                json_file_path = os.path.join(subdir, json_file)
                txt_file_path = os.path.join(subdir, txt_file)
                # Check if both files exist
                if os.path.exists(json_file_path) and os.path.exists(txt_file_path):
                    # If both files exist, add the redundant directory to the list
                    print(f"{current_dir_path} is to be deleted")
                    directories_to_delete.append(current_dir_path)

    # Delete the collected directories
    for dir_path in directories_to_delete:
        shutil.rmtree(dir_path)
        print(f"Deleted redundant directory: {dir_path}")

def clean_mp3_dirs(directory):
    clean_fullwidth_characters(directory)
    move_remaining_mp3_to_their_subdirs()
    merge_directories(directory)
    delete_mp3_if_text_or_json_exists(directory)


import os
import shutil


def del_wrong_subdirs(root_dir):
    # Define the expected maximum directory depth
    expected_max_depth = 10  # Based on home/user/PycharmProjects/rag/datasets/evaluation_data/diarized_youtube_content_2023-10-06/<channel_name>/<release_date>_<video_title>/

    for subdir, dirs, files in os.walk(root_dir, topdown=False):
        # Split the path to evaluate its depth
        path_parts = subdir.split(os.sep)

        # Check if the directory name contains '_diarized_content' or '_diarized_content_processed_diarized'
        if '_diarized_content' in subdir or '_diarized_content_processed_diarized' in subdir:
            # Delete the directory and its content
            # print(f"Removed directory and its content: {subdir}")
            shutil.rmtree(subdir)
        elif len(path_parts) > expected_max_depth:
            # Delete the directory and its content if it exceeds the maximum depth
            print(f"Removed directory and its content: {subdir}")
            shutil.rmtree(subdir)


def copy_and_verify_files():
    # Define the root directory for PycharmProjects
    pycharm_projects_dir = f"{root_directory()}/../"

    # Define the source directories
    csv_source_dir = os.path.join(pycharm_projects_dir, "mev.fyi/data/")
    articles_pdf_source_dir = os.path.join(pycharm_projects_dir, "mev.fyi/data/articles_pdf_download/")
    articles_thumbnails_source_dir = os.path.join(pycharm_projects_dir, "mev.fyi/data/article_thumbnails/")
    research_paper_thumbnails_source_dir = os.path.join(pycharm_projects_dir, "mev.fyi/data/research_papers_pdf_thumbnails/")
    papers_pdf_source_dir = os.path.join(pycharm_projects_dir, "mev.fyi/data/papers_pdf_downloads/")

    # Define the destination directories
    csv_destination_dir = os.path.join(pycharm_projects_dir, "rag/datasets/evaluation_data/")
    articles_pdf_destination_dir = os.path.join(pycharm_projects_dir, "rag/datasets/evaluation_data/articles_2023-12-05/")
    articles_thumbnails_destination_dir = os.path.join(pycharm_projects_dir, "rag_app_vercel/app/public/research_paper_thumbnails/")
    papers_pdf_thumbnails_destination_dir = os.path.join(pycharm_projects_dir, "rag_app_vercel/app/public/research_paper_thumbnails/")
    # articles_thumbnails_destination_dir = os.path.join(pycharm_projects_dir, "app/public/research_paper_thumbnails/")
    # papers_pdf_thumbnails_destination_dir = os.path.join(pycharm_projects_dir, "app/public/research_paper_thumbnails/")
    papers_pdf_destination_dir = os.path.join(pycharm_projects_dir, "rag/datasets/evaluation_data/baseline_evaluation_research_papers_2023-11-21/")

    # List of CSV files to copy
    csv_files_to_copy = [
        "paper_details.csv",
        "links/articles_updated.csv",
        "links/youtube/youtube_videos.csv",
        "links/youtube/youtube_channel_handles.txt"
    ]

    # Create the destination directories if they do not exist
    os.makedirs(csv_destination_dir, exist_ok=True)
    os.makedirs(articles_pdf_destination_dir, exist_ok=True)
    os.makedirs(papers_pdf_destination_dir, exist_ok=True)
    os.makedirs(articles_thumbnails_destination_dir, exist_ok=True)

    # Copy and verify CSV files
    for file_name in csv_files_to_copy:
        source_file = os.path.join(csv_source_dir, file_name)
        destination_file = os.path.join(csv_destination_dir, file_name.split('/')[-1])  # Get the last part if there's a path included
        copy_and_verify(source_file, destination_file)

    # Copy PDF files without size verification
    copy_all_files(articles_pdf_source_dir, articles_pdf_destination_dir)
    copy_all_files(papers_pdf_source_dir, papers_pdf_destination_dir)
    copy_all_files(articles_thumbnails_source_dir, articles_thumbnails_destination_dir, file_extension='.png')
    copy_all_files(research_paper_thumbnails_source_dir, papers_pdf_thumbnails_destination_dir, file_extension='.png')

    copy_and_rename_website_docs_pdfs()
    print("File copying completed.")


def copy_and_verify(source_file, destination_file):
    try:
        # Verify file size before copying
        if os.path.exists(destination_file):
            source_size = os.path.getsize(source_file)
            destination_size = os.path.getsize(destination_file)

            if destination_size > source_size:
                # raise ValueError(f"File {os.path.basename(source_file)} in destination is larger than the source. Copy aborted.")
                print(f"/!\File {os.path.basename(source_file)} in destination is larger than the source. Copy aborted.")
                return

        shutil.copy(source_file, destination_file)
        # print(f"Copied: {source_file} to {destination_file}")
    except IOError as e:
        print(f"Unable to copy file. {e}")
    except ValueError as e:
        print(e)
        # Stop the process if size condition is not met
    except Exception as e:
        print(f"Unexpected error: {e}")


def copy_all_files(source_dir, destination_dir, file_extension='.pdf'):
    for file_name in os.listdir(source_dir):
        if file_name.lower().endswith(file_extension):  # Ensuring it is a PDF file
            source_file = os.path.join(source_dir, file_name)
            destination_file = os.path.join(destination_dir, file_name)
            try:
                shutil.copy(source_file, destination_file)
                # print(f"Copied: {source_file} to {destination_file}")
            except IOError as e:
                print(f"Unable to copy file. {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")


def copy_and_rename_website_docs_pdfs():
    root_dir = root_directory()
    source_directories = {
        f'{root_dir}/../mev.fyi/data/flashbots_docs_pdf': f'{root_dir}/datasets/evaluation_data/flashbots_docs_2024_01_07',
        f'{root_dir}/../mev.fyi/data/ethereum_org_website_content': f'{root_dir}/datasets/evaluation_data/ethereum_org_content_2024_01_07'
    }

    for source_root, target_root in source_directories.items():
        # Ensure the target directory exists
        os.makedirs(target_root, exist_ok=True)

        # Walk through the source directory
        for root, dirs, files in os.walk(source_root):
            for file in files:
                if file.endswith(('.pdf', '.pdfx')):
                    # Construct the relative path
                    relative_path = os.path.relpath(root, source_root)
                    # Replace directory separators with '-' and remove leading directory name if present
                    leading_dir_name = os.path.basename(source_root) + '-'
                    relative_path = relative_path.replace(os.path.sep, '-')
                    if relative_path == '.':
                        new_filename = file
                    elif relative_path.startswith(leading_dir_name):
                        new_filename = relative_path[len(leading_dir_name):] + '-' + file
                    else:
                        new_filename = relative_path + '-' + file

                    # Change the file extension from .pdfx to .pdf if necessary
                    if new_filename.endswith('.pdfx'):
                        new_filename = new_filename[:-1]

                    # Construct the full source and target paths
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(target_root, new_filename)

                    # Copy the file
                    shutil.copy2(source_file, target_file)
                    print(f"Copied and renamed {source_file.split('/')[-1]} to {target_file.split('/')[-1]}")


if __name__ == '__main__':
    pass
    copy_and_verify_files()

    # copy_and_rename_website_docs_pdfs()

    # directory = f"{root_directory()}/datasets/evaluation_data/diarized_youtube_content_2023-10-06"
    # clean_fullwidth_characters(directory)
    # move_remaining_mp3_to_their_subdirs()
    # merge_directories(directory)
    # delete_mp3_if_text_or_json_exists(directory)

    # directory = f"{root_directory()}/datasets/evaluation_data/diarized_youtube_content_2023-10-06"
    # pdf_dir = f"{root_directory()}/datasets/evaluation_data/baseline_evaluation_research_papers_2023-10-05"
    # # clean_mp3_dirs(directory=directory)
    # del_wrong_subdirs(directory)
    # move_remaining_txt_to_their_subdirs()
    # move_remaining_json_to_their_subdirs()
    # print_frontend_content()
    # delete_mp3_if_text_or_json_exists(directory)
    # save_data_into_zip()
    # copy_txt_files_to_transcripts()

