import concurrent.futures
import logging
import os
from functools import partial
from pathlib import Path
from typing import Union, Callable

from llama_hub.file.pymu_pdf.base import PyMuPDFReader

from src.Llama_index_sandbox import root_dir
from src.Llama_index_sandbox.constants import *
from src.Llama_index_sandbox.data_ingestion_pdf.utils import is_valid_title, flashbots_title_extraction, \
    ethereum_org_title_extraction, extract_author_and_release_date_ethereum_org, \
    extract_author_and_release_date_flashbots, extract_title, extract_link, extract_author_and_release_date
from src.Llama_index_sandbox.utils.utils import timeit


def load_single_pdf(file_path, title_extraction_func: Callable, extract_author_and_release_date_func: Callable, author: str, release_date: str, pdf_link: str, loader=PyMuPDFReader(),
                    debug=False):
    try:
        title = extract_title(file_path, title_extraction_func)
        if title is not None:
            link = extract_link(domain_url=pdf_link, search_query=title)
            extracted_author, extracted_release_date = extract_author_and_release_date(link=link, extract_author_and_release_date_func=extract_author_and_release_date_func)
        else:
            link, extracted_author, extracted_release_date = None, None, None
            logging.warning(f"Couldn't find title for [{file_path}]")

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
                'authors': extracted_author if extracted_author is not None else author,
                'pdf_link': link if link is not None else pdf_link,
                'release_date': extracted_release_date if extracted_release_date is not None else release_date,
            })

        if debug:
            logging.info(f'Processed [{title}]')
        return documents
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        return []


@timeit
def load_pdfs(directory_path: Union[str, Path], title_extraction_func: Callable, extract_author_and_release_date_func: Callable, author: str, release_date: str, pdf_link: str, num_files: int = None, num_cpus: int = None, debug=False):
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    all_documents = []
    partial_load_single_pdf = partial(load_single_pdf, title_extraction_func=title_extraction_func, extract_author_and_release_date_func=extract_author_and_release_date_func,
                                      author=author, release_date=release_date, pdf_link=pdf_link, debug=debug)

    files_gen = directory_path.glob("*.pdf")
    files = [next(files_gen) for _ in range(num_files)] if num_files is not None else list(files_gen)

    pdf_loaded_count = 0  # Initialize the counter

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpus) as executor:
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


def load_docs_as_pdf(debug=False, num_files: int = None, num_cpus: int = None):
    # Configuration for the PDF processing
    config = {
        'datasets/evaluation_data/ethereum_org_content_2024_01_07': {
            'title_extraction_func': ethereum_org_title_extraction,
            'extract_author_and_release_date_func': extract_author_and_release_date_ethereum_org,
            'author': 'Ethereum.org',
            'pdf_link': 'https://ethereum.org/',
            'release_date': ''
        },
        'datasets/evaluation_data/flashbots_docs_2024_01_07': {
            'title_extraction_func': flashbots_title_extraction,
            'extract_author_and_release_date_func': extract_author_and_release_date_flashbots,
            'author': 'Flashbots Docs',
            'pdf_link': 'https://docs.flashbots.net/',
            'release_date': ''
        }
    }

    all_docs = []
    for directory, details in config.items():
        directory_path = os.path.join(root_dir, directory)
        title_extraction_func = details['title_extraction_func']
        extract_author_and_release_date_func = details['extract_author_and_release_date_func']
        author = details['author']
        release_date = details['release_date']
        pdf_link = details['pdf_link']

        logging.info(f"Processing directory: {directory_path}")
        all_docs += load_pdfs(directory_path, title_extraction_func, extract_author_and_release_date_func, author, release_date, pdf_link, debug=debug,
                              num_files=num_files, num_cpus=num_cpus)
    return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    num_cpus = os.cpu_count()# 1
    load_docs_as_pdf(debug=True, num_cpus=num_cpus)
