import concurrent.futures
import logging
import os
from functools import partial
from pathlib import Path
from typing import Union, Callable

import pandas as pd
import numpy as np
from llama_hub.file.pymu_pdf.base import PyMuPDFReader

from src.Llama_index_sandbox import root_dir
from src.Llama_index_sandbox.constants import *
from src.Llama_index_sandbox.data_ingestion_pdf.utils import is_valid_title, flashbots_title_extraction, \
    ethereum_org_title_extraction, extract_author_and_release_date_ethereum_org, \
    extract_author_and_release_date_flashbots, extract_title, extract_link, extract_author_and_release_date
from src.Llama_index_sandbox.utils.utils import timeit


def sanitize_metadata_value(value):
    """Sanitize metadata value to ensure it's not None or np.nan."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    return str(value).replace('"', '')


def load_single_pdf(file_path, title_extraction_func: Callable, extract_author_and_release_date_func: Callable, author: str, release_date: str, pdf_link: str, existing_metadata: pd.DataFrame, loader=PyMuPDFReader(),
                    debug=False):
    try:
        filename = os.path.basename(file_path)
        existing_row = existing_metadata[existing_metadata['document_name'] == filename]

        if not existing_row.empty:
            documents_details = existing_row.to_dict('records')[0]
            title = sanitize_metadata_value(documents_details['title'])
            extracted_author = sanitize_metadata_value(documents_details['authors'])
            link = sanitize_metadata_value(documents_details['pdf_link'])
            extracted_release_date = sanitize_metadata_value(documents_details['release_date'])
        else:
            title = sanitize_metadata_value(extract_title(file_path, title_extraction_func))
            if title:
                link = sanitize_metadata_value(extract_link(domain_url=pdf_link, search_query=title))
                extracted_author, extracted_release_date = extract_author_and_release_date(link=link, extract_author_and_release_date_func=extract_author_and_release_date_func)
                extracted_author = sanitize_metadata_value(extracted_author)
                extracted_release_date = sanitize_metadata_value(extracted_release_date)
            else:
                link, extracted_author, extracted_release_date = '', '', ''
                logging.warning(f"Couldn't find title for [{file_path}]")

        # Check if the title is valid
        if not is_valid_title(title):
            logging.warning(f"Skipping file with invalid title: {title} in {file_path}")
            return [], {}

        documents = loader.load(file_path=file_path)

        for document in documents:
            if 'file_path' in document.metadata.keys():
                del document.metadata['file_path']

            if document.text is None or document.text == '':
                logging.error(f"Empty content in {title} with {file_path}")

            document.metadata.update({
                'document_type': DOCUMENT_TYPES.ARTICLE.value,
                'title': title,
                'authors': extracted_author,
                'pdf_link': link,
                'release_date': extracted_release_date
            })

        if debug:
            logging.info(f'Processed [{filename}] with named [{title}]')
        documents_details = {
            'title': title,
            'authors': extracted_author,
            'pdf_link': link,
            'release_date': extracted_release_date,
            'document_name': filename
        }
        return documents, documents_details
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        return [], {}


@timeit
def load_pdfs(directory_path: Union[str, Path], title_extraction_func: Callable, extract_author_and_release_date_func: Callable, author: str, release_date: str, pdf_link: str, existing_metadata: pd.DataFrame, num_files: int = None, num_cpus: int = None, debug=False):
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    all_documents = []
    partial_load_single_pdf = partial(load_single_pdf, title_extraction_func=title_extraction_func, extract_author_and_release_date_func=extract_author_and_release_date_func,
                                      author=author, release_date=release_date, pdf_link=pdf_link, existing_metadata=existing_metadata, debug=debug)

    files_gen = directory_path.glob("*.pdf")
    files = [next(files_gen) for _ in range(num_files)] if num_files is not None else list(files_gen)

    pdf_loaded_count = 0  # Initialize the counter
    # Initialize a list to accumulate metadata
    metadata_accumulator = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpus) as executor:
        futures = {executor.submit(partial_load_single_pdf, file_path=pdf_file): pdf_file for pdf_file in files}

        for future in concurrent.futures.as_completed(futures):
            pdf_file = futures[future]
            try:
                documents, documents_details = future.result()
                if documents_details['title'] not in [md['title'] for md in metadata_accumulator]:
                    metadata_accumulator.append(documents_details)
                    pdf_loaded_count += 1

                all_documents.extend(documents)
                pdf_loaded_count += 1  # Increment the counter for each successful load
            except Exception as e:
                logging.info(f"Failed to process {pdf_file}, with reason: {e}")
                # os.remove(pdf_file)

    logging.info(f"Successfully loaded [{pdf_loaded_count}] documents from [{author}].")
    return all_documents, metadata_accumulator


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
    all_metadata = []
    # Load existing dataframe if it exists
    csv_path = os.path.join(root_dir, 'datasets/evaluation_data/docs_details.csv')
    if os.path.exists(csv_path):
        existing_metadata = pd.read_csv(csv_path)
    else:
        existing_metadata = pd.DataFrame(columns=['title', 'authors', 'pdf_link', 'release_date', 'document_name'])

    for directory, details in config.items():
        directory_path = os.path.join(root_dir, directory)
        title_extraction_func = details['title_extraction_func']
        extract_author_and_release_date_func = details['extract_author_and_release_date_func']
        author = details['author']
        release_date = details['release_date']
        pdf_link = details['pdf_link']

        logging.info(f"Processing directory: {directory_path}")
        all_documents, all_documents_details = load_pdfs(directory_path, title_extraction_func, extract_author_and_release_date_func, author, release_date, pdf_link, existing_metadata=existing_metadata, debug=debug,
                              num_files=num_files, num_cpus=num_cpus)
        all_docs += all_documents
        all_metadata += all_documents_details

    # Save to CSV
    df = pd.DataFrame(all_metadata)
    csv_path = os.path.join(root_dir, 'datasets/evaluation_data/docs_details.csv')
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['title'])
    else:
        combined_df = df

    combined_df.to_csv(csv_path, index=False)
    return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    num_cpus = 1  # os.cpu_count()# 1
    load_docs_as_pdf(debug=True, num_cpus=num_cpus)
