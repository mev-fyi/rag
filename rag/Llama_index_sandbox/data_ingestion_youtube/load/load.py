import concurrent.futures
import logging
import os
from functools import partial
from pathlib import Path
from typing import Union
import pandas as pd
from llama_index import SimpleDirectoryReader
import re

from rag.Llama_index_sandbox import root_dir
from rag.Llama_index_sandbox.utils import timeit


def load_single_video_transcript(youtube_videos_df, file_path):
    # Process the title from the file path
    title = str(os.path.basename(file_path).replace('_diarized_content_processed_diarized.txt', '')).split('_')[1].strip()

    # Ensure any sequence of more than one space in title is replaced with a single space
    title = re.sub(' +', ' ', title)

    # Similarly, replace sequences of spaces in the DataFrame's 'title' column
    youtube_videos_df['title'] = youtube_videos_df['title'].str.replace(' +', ' ', regex=True)

    # Now look for a match
    video_row = youtube_videos_df[youtube_videos_df['title'] == title]

    if video_row.empty:
        # logging.info(f"Could not find video transcript for {title}. Passing.")
        return []

    reader = SimpleDirectoryReader(
        input_files=[file_path]
    )
    # NOTE 2023-10-04: .pdf reader creates many documents while .txt from SimpleDirectoryReader
    #  expectedly creates a single document. which one has the correct behavior? do we care?
    documents = reader.load_data()

    # Update 'file_path' metadata and add additional metadata
    for document in documents:
        if 'file_path' in document.metadata.keys():
            del document.metadata['file_path']

        # Update metadata
        document.metadata.update({
            # TODO 2023-10-04: is there an impact of different metadata keys across documents?
            #  Necessarily, multi-document agents deal with several document types?
            'title': video_row.iloc[0]['title'],
            'channel_name': video_row.iloc[0]['channel_name'],
            'video_link': video_row.iloc[0]['url'],
            'published_date': video_row.iloc[0]['published_date']
        })
        # TODO 2023-10-05: how do i explictly tell the document type as video? should i store the youtube transcripts as a separate index?
        #       (1) i would want to avoid the case where the agent only looks as paper index
        #       (2) on the other hand i want the agent to quickly reference video content if it is specifically asked for
        # TODO 2023-09-27: add relevance score as metadata. The score will be highest for research papers, ethresear.ch posts.
        #   It will be high (highest too? TBD.) for talks and conferences in YouTube video_transcript format
        #   It will be relatively lower for podcasts, tweets, and less formal content.
    return documents


@timeit
def load_video_transcripts(directory_path: Union[str, Path]):
    # Convert directory_path to a Path object if it is not already
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    all_documents = []
    videos_path = f"{root_dir}/datasets/evaluation_data/youtube_videos.csv"

    youtube_videos_df = pd.read_csv(videos_path)
    partial_load_single_transcript = partial(load_single_video_transcript, youtube_videos_df=youtube_videos_df)
    video_transcripts_loaded_count = 0

    # Using ThreadPoolExecutor to load PDFs in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map over all video transcript files in the directory
        futures = {executor.submit(partial_load_single_transcript, file_path=video_transcript): video_transcript for video_transcript in directory_path.rglob("*.txt")}

        for future in concurrent.futures.as_completed(futures):
            video_transcript = futures[future]
            try:
                documents = future.result()
                # if documents is an empty list then continue
                if not documents:
                    continue
                all_documents.extend(documents)
                video_transcripts_loaded_count += 1
            except Exception as e:
                logging.info(f"Failed to process {video_transcript}, passing: {e}")
                pass
    logging.info(f"Successfully loaded {video_transcripts_loaded_count} documents from video transcripts.")
    return all_documents
