import concurrent.futures
import logging
import os
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Union, Callable, Dict

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import requests

from llama_hub.file.pymu_pdf.base import PyMuPDFReader

from src.Llama_index_sandbox import root_dir
from src.Llama_index_sandbox.constants import *
from src.Llama_index_sandbox.data_ingestion_pdf.utils import return_driver
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


def extract_link(domain_url: str, search_query: str):
    """ Retrieves the URL of the page after performing a search on the domain URL. """
    driver = return_driver()

    try:
        driver.get(domain_url)
        time.sleep(2)  # Wait for the page to load

        # Open the search bar with CTRL+K
        webdriver.ActionChains(driver).key_down(Keys.CONTROL).send_keys('k').key_up(Keys.CONTROL).perform()
        time.sleep(0.5)  # Wait for search bar to open

        # Find the search input and enter the search query
        search_input = driver.find_element(By.CSS_SELECTOR, "input[type='search']")
        search_input.send_keys(search_query)
        search_input.send_keys(Keys.ENTER)

        # Wait a moment for the page to update
        time.sleep(1)

        # Get the current URL after the search
        result_url = driver.current_url

        return result_url
    except Exception as e:
        print(f"Error extracting link: {e}")
        return None
    finally:
        driver.quit()


def extract_author_and_release_date_ethereum_org(link: str):
    try:
        response = requests.get(link)
        response.raise_for_status()  # Ensure the request was successful

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract release date
        release_date_selector = '#main-content > div > div > div > div > article > div.css-1ac81qj > div.css-1xm9et8 > div.chakra-skeleton.css-1cyydwu > p > a'
        release_date_element = soup.select_one(release_date_selector)
        if not release_date_element:
            # Try alternative selector
            release_date_selector = '#main-content > div > div > div > div > div > p'
            release_date_element = soup.select_one(release_date_selector)

        release_date = None
        if release_date_element:
            release_date_str = release_date_element.get_text().replace('Page last updated: ', '')
            release_date = datetime.strptime(release_date_str, '%B %d, %Y').strftime('%Y-%m-%d')

        # Extract author
        author_selector = '#main-content > div > div > div > div > article > div.css-1ac81qj > div.css-1xm9et8 > div.chakra-skeleton.css-1tfhr0e > p > a'
        author_element = soup.select_one(author_selector)
        author = author_element.get_text() if author_element else None

        return author, release_date
    except Exception as e:
        print(f"Error extracting author and release date: {e}")
        return None, None

def extract_author_and_release_date_flashbots(link: str):
    try:
        response = requests.get(link)
        response.raise_for_status()  # Ensure the request was successful

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract release date
        release_date_selector = '#main-content > div > div > div > div > article > div.css-1ac81qj > div.css-1xm9et8 > div.chakra-skeleton.css-1cyydwu > p > a'
        release_date_element = soup.select_one(release_date_selector)
        if not release_date_element:
            # Try alternative selector
            release_date_selector = '#main-content > div > div > div > div > div > p'
            release_date_element = soup.select_one(release_date_selector)

        release_date = None
        if release_date_element:
            release_date_str = release_date_element.get_text().replace('Page last updated: ', '')
            release_date = datetime.strptime(release_date_str, '%B %d, %Y').strftime('%Y-%m-%d')

        # Extract author
        author_selector = '#main-content > div > div > div > div > article > div.css-1ac81qj > div.css-1xm9et8 > div.chakra-skeleton.css-1tfhr0e > p > a'
        author_element = soup.select_one(author_selector)
        author = author_element.get_text() if author_element else None

        return author, release_date
    except Exception as e:
        print(f"Error extracting author and release date: {e}")
        return None, None

def extract_author_and_release_date(link, extract_author_and_release_date_func):
    """Custom function to extract the title from the PDF file."""
    return extract_author_and_release_date_func(link)

def load_single_pdf(file_path, title_extraction_func: Callable, extract_author_and_release_date_func: Callable, author: str, release_date: str, pdf_link: str, loader=PyMuPDFReader(),
                    debug=False):
    try:
        title = extract_title(file_path, title_extraction_func)
        link = extract_link(domain_url=pdf_link, search_query=title)
        extracted_author, extracted_release_date = extract_author_and_release_date(link=link, extract_author_and_release_date_func=extract_author_and_release_date_func)

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
            'author'
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
    num_cpus = 1
    load_docs_as_pdf(debug=True, num_cpus=num_cpus)
