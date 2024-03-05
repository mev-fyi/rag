import concurrent.futures
import logging
import os
from functools import partial
from typing import Union, Callable, Tuple

import pandas as pd
import numpy as np
from src.Llama_index_sandbox.custom_pymupdfreader.base import PyMuPDFReader
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.Llama_index_sandbox import root_dir
from src.Llama_index_sandbox.constants import *
from src.Llama_index_sandbox.data_ingestion_pdf.utils import is_valid_title, flashbots_title_extraction, \
    ethereum_org_title_extraction, extract_author_and_release_date_ethereum_org, \
    extract_author_and_release_date_flashbots, extract_title, extract_link, extract_author_and_release_date
from src.Llama_index_sandbox.utils.utils import timeit, save_successful_load_to_csv


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
        # Save the successful load details to CSV
        save_successful_load_to_csv(documents_details, csv_filename='docs.csv', fieldnames=['title', 'authors', 'pdf_link', 'release_date', 'document_name'])
        return documents, documents_details
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        return [], {}


@timeit
def load_pdfs(directory_path: Union[str, Path], title_extraction_func: Callable, extract_author_and_release_date_func: Callable, author: str, release_date: str, pdf_link: str, files, existing_metadata: pd.DataFrame, num_files: int = None, num_cpus: int = None, debug=False):
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    if num_files is not None:
        files = files[:num_files]

    all_documents = []
    partial_load_single_pdf = partial(load_single_pdf, title_extraction_func=title_extraction_func, extract_author_and_release_date_func=extract_author_and_release_date_func,
                                      author=author, release_date=release_date, pdf_link=pdf_link, existing_metadata=existing_metadata, debug=debug)

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


def check_file_exclusion(file_path: Path, title_extraction_func: Callable, exclude_titles: set, exclude_filenames: set) -> Union[Path, None]:
    """
    Check if a file should be excluded based on its title or filename.

    Returns the file path if it should be included, None otherwise.
    """
    filename = file_path.name.lower()  # Convert filename to lowercase
    title = sanitize_metadata_value(extract_title(file_path, title_extraction_func))

    # Check for exclusion based on title and filename
    if not any(excluded_title.lower() in title.lower() for excluded_title in exclude_titles) and \
            not any(excluded_filename.lower() in filename for excluded_filename in exclude_filenames):
        return file_path
    else:
        logging.info(f"[Exclusion] Excluding file: [{filename}] with title: [{title}]")
        return None


@timeit
def load_docs_as_pdf(debug=False, num_files: int = None, num_cpus: int = None):
    # Configuration for the PDF processing
    config = {
        'datasets/evaluation_data/ethereum_org_content_2024_01_07': {
            'title_extraction_func': ethereum_org_title_extraction,
            'extract_author_and_release_date_func': extract_author_and_release_date_ethereum_org,
            'author': 'Ethereum.org',
            'pdf_link': 'https://ethereum.org/',
            'release_date': '',
            'exclude_titles': [
                'Content standardization',
                'Style Guide',
                'How can I get involved?',
                'Adding a quiz',
                '"The Graph: Fixing Web3 data querying"',
                'Translation Program lang: en',
                'Content buckets lang: en',
                'Adding developer tools lang: en',
                'Contributing',
                'Language resources',
                'Code of conduct',
                'How can I get involved?',
                'Online communities',
                'About Us',
                'The Graph: Fixing Web3 data querying',
            ],  # List of titles to exclude
            'exclude_filenames': ['contributing']  # List of filenames to exclude
        },
        'datasets/evaluation_data/flashbots_docs_2024_01_07': {
            'title_extraction_func': flashbots_title_extraction,
            'extract_author_and_release_date_func': extract_author_and_release_date_flashbots,
            'author': 'Flashbots Docs',
            'pdf_link': 'https://docs.flashbots.net/',
            'release_date': '',
            'exclude_titles': [
                'Join Flashbots',
                'Contributing',
                'Prohibited Use Policy',
                'Terms of Service',
                'Welcome to Flashbots hide_title: true description: The home page of the knowledge base keywords: - flashbots -',
                'Code of Conduct',
                'js hint: calldata | contract_address | function_selector | logs | hash | undefined',

            ],  # List of titles to exclude
            'exclude_filenames': ['policies']  # List of filenames to exclude
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
        # Pre-filter existing metadata based on exclude_titles and exclude_filenames
        exclude_titles = set(details.get('exclude_titles', []))
        exclude_filenames = set(details.get('exclude_filenames', []))
        filtered_metadata = existing_metadata[
            ~(existing_metadata['title'].fillna('').str.lower().isin([t.lower() for t in exclude_titles])) &
            ~(existing_metadata['document_name'].fillna('').str.lower().apply(
                lambda x: any(f.lower() in x for f in exclude_filenames)))
            ]
        filtered_metadata.to_csv(csv_path, index=False)

        directory_path = os.path.join(root_dir, directory)
        # files_gen = Path(directory_path).glob("*.pdf")
        files = []
        directory_path = os.path.join(root_dir, directory)
        # Convert generator to list for parallel processing
        files_gen = list(Path(directory_path).glob("*.pdf"))

        # Apply num_files limitation BEFORE the exclusion checks
        if num_files is not None:
            files_gen = files_gen[:num_files]

        # Prepare function call
        partial_check_file_exclusion = partial(check_file_exclusion,
                                               title_extraction_func=details['title_extraction_func'],
                                               exclude_titles=set(details.get('exclude_titles', [])),
                                               exclude_filenames=set(details.get('exclude_filenames', [])))

        # Initialize ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            # Submit tasks
            future_to_file = {executor.submit(partial_check_file_exclusion, file_path): file_path for file_path in files_gen}

            # Collect results
            files = []
            for future in as_completed(future_to_file):
                result = future.result()
                if result is not None:
                    files.append(result)

        logging.info(f"Processing directory: {directory_path} with {len(files)} files")
        all_documents, all_documents_details = load_pdfs(directory_path, details['title_extraction_func'],
                                                         details['extract_author_and_release_date_func'],
                                                         details['author'], details['release_date'],
                                                         details['pdf_link'], files,
                                                         existing_metadata=filtered_metadata,
                                                         num_files=num_files, num_cpus=num_cpus, debug=debug)
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
