import logging

from llama_index.tools import ToolOutput
from pydantic import Field


class CustomToolOutput(ToolOutput):
    all_formatted_metadata: str = Field(default="")  # Declare the new field with a default value

    def __init__(self, **data):
        super().__init__(**data)

        # Note: Be cautious about changing attributes directly, Pydantic models are not designed for that.
        # We might want to use `self.copy(update={"field": new_value})` instead.

        # Ensure that raw_output has the structure we expect.
        if self.raw_output and hasattr(self.raw_output, 'metadata'):
            self.all_formatted_metadata = format_metadata(self.raw_output)  # Store formatted metadata.
        else:
            self.all_formatted_metadata = "No metadata available"
            print("Warning: raw_output may not contain metadata as expected.")

    def __str__(self) -> str:
        """Provide a basic string representation."""
        msg = f"{self.content}"  # \n\nFetched based on the following sources: \n{self.all_formatted_metadata}\n"
        return msg

    def get_formatted_metadata(self) -> str:
        """A method specifically for retrieving formatted metadata."""
        return self.all_formatted_metadata


def format_metadata(response):
    title_to_metadata = {}

    for node in response.source_nodes:
        meta_info = node.metadata
        score = node.score
        title = meta_info.get('title', 'N/A')
        is_video = 'channel_name' in meta_info

        # Initialize metadata dictionary if not exists
        if title not in title_to_metadata:
            title_to_metadata[title] = {
                'pdf_link': meta_info.get('pdf_link', 'N/A'),
                'release_date': meta_info.get('release_date', 'N/A'),
                'channel_name': meta_info.get('channel_name', 'N/A'),
                'video_link': meta_info.get('video_link', 'N/A'),
                'published_date': meta_info.get('release_date', 'N/A'),
                'chunks_count': 0,
                'highest_score': score,
                'is_video': is_video,
                'formatted_authors': None if is_video else ''
            }

        metadata = title_to_metadata[title]

        # Update authors for non-video sources
        if not is_video and 'authors' in meta_info:
            authors_list = meta_info['authors'].split(', ')
            metadata['formatted_authors'] = ', '.join(authors_list) if authors_list != ['N/A'] else None

        # Increment chunks count and update the highest score
        metadata['chunks_count'] += 1
        metadata['highest_score'] = max(metadata['highest_score'], score)

    # Sorting and formatting metadata
    formatted_metadata_list = [
        f"[Title]: {title}, "
        +
        (
         f"[Channel name]: {meta['channel_name']}, [Video Link]: {meta['video_link']}, [Published date]: {meta['published_date']}, "
         if meta['is_video'] else
         f"[Authors]: {meta['formatted_authors']}, [Link]: {meta['pdf_link']}, [Release date]: {meta['release_date']}, "
        )
        +
        f"[Highest Score]: {meta['highest_score']}"
        for title, meta in sorted(title_to_metadata.items(), key=lambda x: (x[1].get('num_chunks', 0), x[1]['highest_score']), reverse=True)
    ]

    return '\n'.join(formatted_metadata_list)


def log_and_store(store_response_fn, query_str, response, chatbot: bool):
    all_formatted_metadata = format_metadata(response)

    if chatbot:
        msg = f"The answer to the question {query_str} is: \n{response}\n\nFetched based on the following sources: \n{all_formatted_metadata}\n"
    else:
        msg = f"The answer to [{query_str}] is: \n\n```\n{response}\n\n\nFetched based on the following sources/content: \n{all_formatted_metadata}\n```"
        logging.info(f"[Shown to client] {msg}")
    return msg, all_formatted_metadata
    # store_response_fn(query_str, response)
