import time
import logging
import os
from datetime import datetime
from functools import wraps

from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials


import os
import subprocess

from llama_index.llms import ChatMessage, MessageRole


def root_directory() -> str:
    """
    Determine the root directory of the project by using the 'git' command first,
    and if that fails, by manually traversing the directories upwards to find a '.git' directory.
    If none are found or an error occurs, raise an exception.

    Returns:
    - str: The path to the root directory of the project.
    """

    # First, try to use the git command to find the root directory
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
    while current_dir != root:
        try:
            if 'src' in os.listdir(current_dir):
                print(f"Found root directory: {current_dir}")
                return current_dir
            current_dir = os.path.dirname(current_dir)
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
    messages = data.get("messages", [])
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


if __name__ == '__main__':
    print_frontend_content()
    #save_data_into_zip()
    # copy_txt_files_to_transcripts()
