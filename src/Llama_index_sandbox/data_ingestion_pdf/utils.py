import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from pdfminer.high_level import extract_text
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By

from src.anyscale_sandbox.utils import root_directory


from selenium import webdriver

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0',
    # Add more user agents if desired...
]

def return_driver():
    # set up Chrome driver options
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-features=IsolateOrigins,site-per-process")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    # NOTE 2024-01-10: hack: use mev.fyi's chrome drivers and all
    CHROME_BINARY_PATH = f'{root_directory()}/../mev.fyi/src/chromium/chrome-linux64/chrome'
    CHROMEDRIVER_PATH = f'{root_directory()}/../mev.fyi/src/chromium/chromedriver-linux64/chromedriver'

    options = webdriver.ChromeOptions()
    options.binary_location = CHROME_BINARY_PATH

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


def extract_author_and_release_date(link, extract_author_and_release_date_func):
    """Custom function to extract the title from the PDF file."""
    return extract_author_and_release_date_func(link)
