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

        if title not in title_to_metadata:
            title_to_metadata[title] = {
                'formatted_authors': None,
                'pdf_link': 'N/A',
                'release_date': 'N/A',
                'channel_name': 'N/A',
                'video_link': 'N/A',
                'published_date': 'N/A',
                'chunks_count': 0,
                'highest_score': score,
                'is_video': 'channel_name' in meta_info  # Check if it's a video
            }

        # Update authors for non-video sources
        if 'authors' in meta_info and not title_to_metadata[title]['formatted_authors'] and not title_to_metadata[title]['is_video']:
            authors_list = meta_info.get('authors', 'N/A').split(', ')
            formatted_authors = ', '.join(authors_list) if authors_list != ['N/A'] else None
            title_to_metadata[title]['formatted_authors'] = formatted_authors

        # Increment chunks count and update highest score
        title_to_metadata[title]['chunks_count'] += 1
        title_to_metadata[title]['highest_score'] = max(title_to_metadata[title]['highest_score'], score)

        # Update other metadata fields if they are not already set
        for field in ['pdf_link', 'release_date', 'channel_name', 'video_link', 'published_date']:
            if title_to_metadata[title][field] == 'N/A':
                title_to_metadata[title][field] = meta_info.get(field, 'N/A')

    # Sorting metadata based on highest score
    sorted_metadata = sorted(title_to_metadata.items(), key=lambda x: x[1]['highest_score'], reverse=True)

    formatted_metadata_list = []
    for title, meta in sorted_metadata:
        if meta['is_video']:
            formatted_metadata = f"[Title]: {title}, [Channel name]: {meta['channel_name']}, [Video Link]: {meta['video_link']}, [Published date]: {meta['release_date']}, [Highest Score]: {meta['highest_score']}"
        else:
            formatted_metadata = f"[Title]: {title}, [Authors]: {meta['formatted_authors']}, [Link]: {meta['pdf_link']}, [Release date]: {meta['release_date']}, [Highest Score]: {meta['highest_score']}"
        formatted_metadata_list.append(formatted_metadata)

    all_formatted_metadata = '\n'.join(formatted_metadata_list)
    return all_formatted_metadata


def log_and_store(store_response_fn, query_str, response, chatbot: bool):
    all_formatted_metadata = format_metadata(response)

    if chatbot:
        msg = f"The answer to the question {query_str} is: \n{response}\n\nFetched based on the following sources: \n{all_formatted_metadata}\n"
    else:
        msg = f"The answer to [{query_str}] is: \n\n```\n{response}\n\n\nFetched based on the following sources/content: \n{all_formatted_metadata}\n```"
        logging.info(f"[Shown to client] {msg}")
    return msg, all_formatted_metadata
    # store_response_fn(query_str, response)
