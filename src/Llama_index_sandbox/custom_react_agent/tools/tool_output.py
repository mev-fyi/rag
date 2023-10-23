import logging

from llama_index.tools import ToolOutput


class CustomToolOutput(ToolOutput):
    def __init__(self, content, raw_output):
        super().__init__(content, raw_output)
        self.all_formatted_metadata = format_metadata(self.raw_output)  # Store formatted metadata.

    def __str__(self) -> str:
        """Provide a basic string representation."""
        msg = f"{self.content}\n\nFetched based on the following sources: \n{self.all_formatted_metadata}\n"
        return msg

    def get_formatted_metadata(self) -> str:
        """A method specifically for retrieving formatted metadata."""
        return self.all_formatted_metadata


def format_metadata(response):
    title_to_metadata = {}

    for key, meta_info in response.metadata.items():
        title = meta_info.get('title', 'N/A')

        if 'authors' in meta_info:
            authors_list = meta_info.get('authors', 'N/A').split(', ')
            # formatted_authors = authors_list[0] + (' et al.' if len(authors_list) > 3 else ', '.join(authors_list[1:]))
            formatted_authors = authors_list[0] + ', ' + ', '.join(authors_list[1:])
        else:
            formatted_authors = None

        if title not in title_to_metadata:
            title_to_metadata[title] = {
                'formatted_authors': formatted_authors,
                'pdf_link': meta_info.get('pdf_link', 'N/A'),
                'release_date': meta_info.get('release_date', 'N/A'),
                'channel_name': meta_info.get('channel_name', 'N/A'),
                'video_link': meta_info.get('video_link', 'N/A'),
                'published_date': meta_info.get('release_date', 'N/A'),
                'chunks_count': 0
            }

        title_to_metadata[title]['chunks_count'] += 1

        # Sorting metadata based on dates (from most recent to oldest)
    sorted_metadata = sorted(title_to_metadata.items(), key=lambda x: (x[1]['release_date'] if x[1]['release_date'] != 'N/A' else x[1]['published_date']), reverse=True)

    formatted_metadata_list = []
    for title, meta in sorted_metadata:
        if meta['formatted_authors']:
            formatted_metadata = f"[Title]: {title}, [Authors]: {meta['formatted_authors']}, [Link]: {meta['pdf_link']}, [Release date]: {meta['release_date']}"  # , [# chunks retrieved]: {meta['chunks_count']}"
        else:
            formatted_metadata = f"[Title]: {title}, [Channel name]: {meta['channel_name']}, [Video Link]: {meta['video_link']}, [Published date]: {meta['published_date']}"  # , [# chunks retrieved]: {meta['chunks_count']}"

        formatted_metadata_list.append(formatted_metadata)

    # Joining all formatted metadata strings with a newline
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
