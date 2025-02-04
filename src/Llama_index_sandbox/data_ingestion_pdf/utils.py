import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Container, Callable, Set, Union

import numpy as np
import requests
from bs4 import BeautifulSoup

from pdfminer.high_level import extract_text
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import logging

from src.Llama_index_sandbox.utils.utils import root_directory


from selenium import webdriver

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0',
    # Add more user agents if desired...
]


def return_driver():

    CHROME_BINARY_PATH = f'{root_directory()}/src/chromium/chrome-linux64/chrome'
    CHROMEDRIVER_PATH = f'{root_directory()}/src/chromium/chromedriver-linux64/chromedriver'

    options = webdriver.ChromeOptions()
    options.binary_location = CHROME_BINARY_PATH

    driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, chrome_options=options)
    return driver


def return_local_ubuntu_driver():

    CHROME_BINARY_PATH = f'{root_directory()}/src/local_chromium/chrome-linux64/chrome'
    CHROMEDRIVER_PATH = f'{root_directory()}/src/local_chromium/chromedriver-linux64/chromedriver'

    options = webdriver.ChromeOptions()
    options.binary_location = CHROME_BINARY_PATH

    driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, chrome_options=options)
    return driver


def return_driver_docker_gce():
    CHROME_BINARY_PATH = f'{root_directory()}/src/chromium/chrome-linux64/chrome'
    CHROMEDRIVER_PATH = f'{root_directory()}/src/chromium/chromedriver-linux64/chromedriver'

    options = webdriver.ChromeOptions()
    options.binary_location = CHROME_BINARY_PATH
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920x1080')
    options.add_argument('--no-zygote')
    options.add_argument('--single-process')

    driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, chrome_options=options)
    return driver


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
    text = extract_text_pymupdf(file_path, page_numbers=[0])
    title_keyword = 'title: '
    if title_keyword in text:
        start = text.find(title_keyword) + len(title_keyword)
        end = text.find('\n', start)
        title = text[start:end].strip()
    else:
        title = text.splitlines()[0].strip() if text.splitlines() else "Unknown Title"
    return clean_title(title)


def suave_title_extraction(file_path):
    text = extract_text_pymupdf(file_path, page_numbers=[0])
    title_keyword = 'title: '
    description_keyword = 'description:'
    hide_title_keyword = 'hide_title:'

    if title_keyword in text:
        start = text.find(title_keyword) + len(title_keyword)
        end = text.find('\n', start)
        title = text[start:end].strip()
    else:
        title = text.splitlines()[0].strip() if text.splitlines() else "Unknown Title"

    # Remove everything on the right side of "description:", including "description" itself
    if description_keyword in title:
        description_index = title.find(description_keyword)
        title = title[:description_index].strip()

    # Remove everything on the right side of "hide_title:", including "hide_title" itself
    if hide_title_keyword in title:
        hide_title_index = title.find(hide_title_keyword)
        title = title[:hide_title_index].strip()

    return clean_title(title)


def ethereum_org_title_extraction(file_path):
    text = extract_text_pymupdf(file_path, page_numbers=[0])
    title_keyword = 'title: '
    if title_keyword in text:
        start = text.find(title_keyword) + len(title_keyword)
        end = text.find(' description:', start)
        title = text[start:end].strip()
    else:
        title = text.splitlines()[0].strip() if text.splitlines() else "Unknown Title"
    return clean_title(title)


def extract_text_pymupdf(pdf_file: str, page_numbers: Optional[Container[int]] = None) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_file)
    text = ""
    if page_numbers is None:
        page_numbers = range(len(doc))
    for num in page_numbers:
        page = doc.load_page(num)
        text += page.get_text()
    doc.close()
    return text


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
        release_date_selector = '#__docusaurus_skipToContent_fallback > div > div > main > div > div > div > div > article > footer > div > div.col.lastUpdated_vwxv > span > b > time'
        release_date_element = soup.select_one(release_date_selector)
        if not release_date_element:
            # Try alternative selector
            release_date_selector = '#main-content > div > div > div > div > div > p'
            release_date_element = soup.select_one(release_date_selector)

        release_date = None
        if release_date_element:
            release_date_str = release_date_element.get_text().replace('Page last updated: ', '')
            release_date = datetime.strptime(release_date_str, '%b %d, %Y').strftime('%Y-%m-%d')

        # NOTE 2024-01-10: there are no authors on flashbot docs
        author = None

        return author, release_date
    except Exception as e:
        print(f"Error extracting author and release date: {e}")
        return None, None


def extract_title(file_path, title_extraction_func):
    """Custom function to extract the title from the PDF file."""
    return title_extraction_func(file_path)


def extract_link(domain_url: str, search_query: str):
    """ Retrieves the URL of the page after performing a search on the domain URL. """

    try:
        # TODO 2024-03-06: fix driver not being instantiated
        driver = return_local_ubuntu_driver()
        driver.get(domain_url)
        time.sleep(2)  # Wait for the page to load

        # Open the search bar with CTRL+K
        webdriver.ActionChains(driver).key_down(Keys.CONTROL).send_keys('k').key_up(Keys.CONTROL).perform()
        time.sleep(1.5)  # Wait for search bar to open

        # Find the search input and enter the search query
        search_input = driver.find_element(By.CSS_SELECTOR, "input[type='search']")
        search_input.send_keys(search_query)
        search_input.send_keys(Keys.ENTER)

        # Wait a moment for the page to update
        time.sleep(1.5)

        # Get the current URL after the search
        result_url = driver.current_url

        return result_url
    except Exception as e:
        logging.info(f"Error extracting link: {e}")
        driver.quit()
        return None
    finally:
        driver.quit()


def extract_author_and_release_date(link, extract_author_and_release_date_func):
    """Custom function to extract the title from the PDF file."""
    return extract_author_and_release_date_func(link)


def sanitize_metadata_value(value):
    """
    Sanitizes the metadata value to ensure it's neither None nor np.nan.

    Parameters:
    - value (Union[str, float, None]): The metadata value to sanitize.

    Returns:
    - str: A sanitized string value. If the original value is None or np.nan, it returns an empty string.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    return str(value).replace('"', '')


def check_file_exclusion(file_path: Path, title_extraction_func: Callable, exclude_titles: Set[str], exclude_filenames: Set[str]) -> Union[Path, None]:
    """
    Determines if a file should be excluded based on its title or filename.
    Now with added performance logging to understand slow parts.
    """
    # Ensure excluded titles and filenames are lowercase outside of this function.
    filename = file_path.name.lower()

    # Extract and sanitize the title.
    title = sanitize_metadata_value(extract_title(file_path, title_extraction_func))

    # Check for exclusion based on title and filename.
    excluded_due_to_title = any(excluded_title in title for excluded_title in exclude_titles)
    excluded_due_to_filename = any(excluded_filename in filename for excluded_filename in exclude_filenames)

    if not excluded_due_to_title and not excluded_due_to_filename:
        return file_path
    else:
        return None
