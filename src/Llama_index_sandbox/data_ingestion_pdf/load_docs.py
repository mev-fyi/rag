import concurrent.futures
import logging
import os
from functools import partial
from pathlib import Path
from typing import Union, Callable, Dict

import pandas as pd
from llama_hub.file.pymu_pdf.base import PyMuPDFReader

from src.Llama_index_sandbox import root_dir
from src.Llama_index_sandbox.constants import *
from src.Llama_index_sandbox.utils.utils import timeit
import re

from pdfminer.high_level import extract_text

def clean_title(title: str) -> str:
    """Remove non-breaking spaces and other non-standard whitespace."""
    # Replace non-breaking spaces and other special whitespaces with regular space
    title = re.sub(r'\s+', ' ', title)
    # Strip leading and trailing whitespaces
    return title.strip()

def is_valid_title(title):
    # Add more checks as necessary
    invalid_indicators = ["Unknown Title", "json title", "import ", "{", "}", 'js hint: "calldata"']
    return not any(indicator in title for indicator in invalid_indicators)

def flashbots_title_extraction(file_path):
    text = extract_text(file_path, page_numbers=[0])
    title_keyword = 'title: '
    if title_keyword in text:
        start = text.find(title_keyword) + len(title_keyword)
        end = text.find('\n', start)
        title = text[start:end].strip()
    else:
        title = text.splitlines()[0].strip() if text.splitlines() else "Unknown Title"
    return clean_title(title)

def ethereum_org_title_extraction(file_path):
    text = extract_text(file_path, page_numbers=[0])
    title_keyword = 'title: '
    if title_keyword in text:
        start = text.find(title_keyword) + len(title_keyword)
        end = text.find(' description:', start)
        title = text[start:end].strip()
    else:
        title = text.splitlines()[0].strip() if text.splitlines() else "Unknown Title"
    return clean_title(title)

def extract_title(file_path, title_extraction_func):
    """Custom function to extract the title from the PDF file."""
    return title_extraction_func(file_path)


def load_single_pdf(file_path, title_extraction_func: Callable, author: str, release_date: str, pdf_link: str, loader=PyMuPDFReader(),
                    debug=False):
    try:
        title = extract_title(file_path, title_extraction_func)

        # Check if the title is valid
        if not is_valid_title(title):
            logging.warning(f"Skipping file with invalid title: {title} in {file_path}")
            return []

        documents = loader.load(file_path=file_path)

        for document in documents:
            if 'file_path' in document.metadata.keys():
                del document.metadata['file_path']

            if document.text is None or document.text == '':
                logging.error(f"Empty content in {title} with {file_path}")

            document.metadata.update({
                'document_type': DOCUMENT_TYPES.ARTICLE.value,
                'title': title,
                'authors': author,
                'pdf_link': pdf_link,
                'release_date': release_date,
            })

        if debug:
            logging.info(f'Processed [{title}]')
        return documents
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        return []


@timeit
def load_pdfs(directory_path: Union[str, Path], title_extraction_func: Callable, author: str, release_date: str, pdf_link: str, num_files: int = None, debug=False):
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    all_documents = []
    partial_load_single_pdf = partial(load_single_pdf, title_extraction_func=title_extraction_func, author=author, release_date=release_date, pdf_link=pdf_link, debug=debug)

    files_gen = directory_path.glob("*.pdf")
    files = [next(files_gen) for _ in range(num_files)] if num_files is not None else list(files_gen)

    pdf_loaded_count = 0  # Initialize the counter

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(partial_load_single_pdf, file_path=pdf_file): pdf_file for pdf_file in files}

        for future in concurrent.futures.as_completed(futures):
            pdf_file = futures[future]
            try:
                documents = future.result()
                all_documents.extend(documents)
                pdf_loaded_count += 1  # Increment the counter for each successful load
            except Exception as e:
                logging.info(f"Failed to process {pdf_file}, removing file: {e}")
                os.remove(pdf_file)

    logging.info(f"Successfully loaded [{pdf_loaded_count}] documents.")
    return all_documents


def load_docs_as_pdf(debug=False, num_files: int = None):
    # Configuration for the PDF processing
    config = {
        'datasets/evaluation_data/ethereum_org_content_2024_01_07': {
            'title_extraction_func': ethereum_org_title_extraction,
            'author': 'Ethereum.org',
            'pdf_link': 'https://ethereum.org/',
            'release_date': ''
        },
        'datasets/evaluation_data/flashbots_docs_2024_01_07': {
            'title_extraction_func': flashbots_title_extraction,
            'author': 'Flashbots Docs',
            'pdf_link': 'https://docs.flashbots.net/',
            'release_date': ''
        }
    }

    all_docs = []
    for directory, details in config.items():
        directory_path = os.path.join(root_dir, directory)
        title_extraction_func = details['title_extraction_func']
        author = details['author']
        release_date = details['release_date']
        pdf_link = details['pdf_link']

        logging.info(f"Processing directory: {directory_path}")
        all_docs += load_pdfs(directory_path, title_extraction_func, author, release_date, pdf_link, debug=debug, num_files=num_files)
    return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    load_docs_as_pdf(debug=True)
