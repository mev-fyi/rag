import logging
import os
import re
import time
import uuid

import backoff
import requests
from requests_oauthlib import OAuth1

from src.Llama_index_sandbox import root_dir
from src.Llama_index_sandbox.data_ingestion_pdf.utils import return_driver_docker_gce


def take_screenshot_and_upload(url):
    """
    Takes a screenshot of the given URL and uploads it to Twitter using chunked upload.
    :param url: The URL to take a screenshot of
    :return: The media ID on Twitter or None if failed
    """
    driver = return_driver_docker_gce()
    try:
        driver.get(url)
        time.sleep(3)  # Wait for dynamic content

        # Generate a unique filename
        unique_filename = f"{uuid.uuid4()}.png"
        if not os.path.exists(f"{root_dir}/tmp/"):
            os.makedirs(f"{root_dir}/tmp/")
        screenshot_path = f'{root_dir}/tmp/{unique_filename}'

        driver.save_screenshot(screenshot_path)
        media_id = upload_media_chunked(screenshot_path, 'image/png', 'tweet_image', True)  # Shared picture
        return media_id
    except Exception as e:
        print(f"Failed to take screenshot and upload: {e}")
        return None
    finally:
        driver.quit()
        # Clean up the screenshot file
        if os.path.exists(screenshot_path):
            os.remove(screenshot_path)


def upload_media_chunked(file_path, media_type, media_category, is_shared):
    # OAuth 1.0a authentication
    oauth = OAuth1(
        client_key=os.environ["TWITTER_CONSUMER_KEY"],
        client_secret=os.environ["TWITTER_CONSUMER_SECRET"],
        resource_owner_key=os.environ["TWITTER_ACCESS_TOKEN"],
        resource_owner_secret=os.environ["TWITTER_ACCESS_TOKEN_SECRET"]
    )

    # Initialize the upload
    url = 'https://upload.twitter.com/1.1/media/upload.json'
    init_data = {
        'command': 'INIT',
        'total_bytes': os.path.getsize(file_path),
        'media_type': media_type,
        'media_category': media_category # ,
        # 'shared': str(is_shared).lower()  # Convert boolean to string
    }
    init_response = requests.post(url, data=init_data, auth=oauth)
    if (init_response.status_code != 200) and (init_response.status_code != 202):
        raise Exception(f"INIT request failed: {init_response.text}")
    media_id = init_response.json()['media_id_string']

    # Upload the file in chunks
    with open(file_path, 'rb') as file:
        segment_id = 0
        while True:
            chunk = file.read(4 * 1024 * 1024)  # 4 MB per chunk
            if not chunk:
                break
            append_data = {
                'command': 'APPEND',
                'media_id': media_id,
                'segment_index': segment_id
            }
            append_response = requests.post(url, data=append_data, files={'media': chunk}, auth=oauth)
            if append_response.status_code != 204:  # HTTP 204 indicates success
                raise Exception(f"APPEND request failed: {append_response.text}")
            segment_id += 1

    # Finalize the upload
    finalize_data = {'command': 'FINALIZE', 'media_id': media_id}
    finalize_response = requests.post(url, data=finalize_data, auth=oauth)
    if (finalize_response.status_code != 200) and (finalize_response.status_code != 201):
        raise Exception(f"FINALIZE request failed: {finalize_response.text}")

    return media_id


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5)
def safe_request(url, headers):
    """
    Safely make an HTTP request with retries on failures.
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Request Exception: {e}")
        return None


def split_response_into_tweets(response, username):
    """
    Splits the response into tweet-sized chunks, respecting bullet points, newlines, and URL formatting.
    :param response: The response message to be split
    :param username: The username to whom the reply should be addressed
    :return: A list of tweet-sized chunks
    """
    chunks = []
    current_chunk = ""
    is_first_chunk = True

    bullet_point_pattern = re.compile(r'^\d+\.')
    source_url_pattern = re.compile(r'\((?i)source: <(https?://[^\s]+)>\)')

    # Replace source URL pattern
    response = source_url_pattern.sub(r'(\1)', response)

    def split_long_bullet_point(line):
        """Splits a long bullet point into two at a sentence boundary."""
        sentences = re.split(r'(?<=[.!?]) +', line)
        mid_point = len(sentences) // 2
        return ' '.join(sentences[:mid_point]), ' '.join(sentences[mid_point:])

    # Split the response by newlines to respect paragraph/bullet points
    lines = response.split('\n')
    for i, line in enumerate(lines):
        if bullet_point_pattern.match(line.strip()):
            # Handle long bullet points
            if len(line) > TWEET_CHAR_LENGTH - (2 + len(username) if is_first_chunk else 0):
                first_half, second_half = split_long_bullet_point(line)
                lines.insert(i + 1, second_half)
                line = first_half

        # Check if adding the line exceeds the character limit
        if len(current_chunk + line) + 2 > TWEET_CHAR_LENGTH - (2 + len(username) if is_first_chunk else 0):
            if current_chunk.strip():  # Add chunk if it's not empty
                chunks.append(current_chunk.strip())
                current_chunk = ""
                is_first_chunk = False

        current_chunk += line + "\n"

        # Check for bullet points to break the text or if it's the last line
        if bullet_point_pattern.match(line.strip()) or i == len(lines) - 1:
            if current_chunk.strip():  # Add chunk if it's not empty
                chunks.append(current_chunk.strip())
                current_chunk = ""

    # Cleanup: Remove any empty chunks and source mentions
    chunks = [re.sub(r'source:\s+', '', chunk, flags=re.IGNORECASE) for chunk in chunks if chunk]

    return chunks


TWEET_CHAR_LENGTH = 280
vitalik_ethereum_roadmap_2023 = """
By popular demand, an updated roadmap diagram for 2023! https://t.co/oxo58A2KuG
Here was the one from last year. Notice that it's actually quite similar! As Ethereum's technical path forward continues to solidify, there are relatively few changes. I'll go through the important ones.
https://t.co/F1MpfNdfa7
The role of single slot finality (SSF) in post-Merge PoS improvement is solidifying. It's becoming clear that SSF is the easiest path to resolving a lot of the Ethereum PoS design's current weaknesses.
See:
https://t.co/l8OcVsXCOW
https://t.co/8sM6DieMjg https://t.co/i6Gz36huW4
Significant progress on the Surge (rollup scaling) this year, both on EIP-4844 and from rollups themselves.
https://t.co/uEjd2VETQU continues to be a good page to follow.
Also, cross-rollup standards and interop has been highlighted as an area for long-term improvements. https://t.co/ixUsZEo7pi
The Scourge has been redesigned somewhat. It's now about fighting economic centralization in PoS in general, in two key theaters: (i) MEV, (ii) general stake pooling issues.
See also:
https://t.co/PvdKeYmglL
https://t.co/un6IuyssL4
https://t.co/tvOAh2T1eb https://t.co/tjYiwmzYeB
Significant progress in the Verge; Verkle trees are coming closer to being ready for inclusion.
See: https://t.co/Bg2KXGZzSF
"Increase L1 gas limit" was removed to emphasize that the limit can be raised *at any time*; no need to wait for full SNARKs esp for small increases. https://t.co/92M6JtjBBD
"State expiry" has been shrunk, to reflect a general consensus that it's a fairly low-priority and low-urgency item esp given stateless clients and PBS / execution tickets. https://t.co/Z0ziIsDnGt
VDFs have been shrunk to reflect a temporary reduced emphasis, due to cryptographic weaknesses in existing constructions:
https://t.co/XUfEf9Zlii
Deep crypto (eg. obfuscation) and delay-encrypted mempools have been added to reflect growing research interest in these topics. https://t.co/7cMurJVUfp
Big thanks to @drakefjustin @fradamt @mikeneuder @dankrad @barnabemonnot @asanso @icebearhww @domothy @gballet  for feedback!
@VitalikButerin @drakefjustin @fradamt @mikeneuder @dankrad @barnabemonnot @asanso @icebearhww @domothy @gballet @mevfyi explain thread
"""


def lookup_user_by_username(username):
    # Replace 'your_bearer_token_here' with your actual bearer token.
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")

    # Endpoint URL for the Twitter API v2 user lookup by username
    endpoint_url = f"https://api.twitter.com/2/users/by/username/{username}"

    # Prepare the request headers with authorization
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "User-Agent": "v2UserLookupPython"
    }

    # Make the GET request to the Twitter API
    response = requests.get(endpoint_url, headers=headers)

    # Check the response status code
    if response.status_code == 200:
        user_data = response.json().get("data", {})
        user_id = user_data.get("id")
        logging.info(f"User ID for @{username} is {user_id}")
        return user_id
    else:
        logging.error(f"Error fetching user ID: {response.status_code} - {response.text}")
        return None


def extract_command_and_message(message):
    """
    Extracts the command and the message from the tweet.

    :param message: The message from the tweet
    :return: A tuple containing the command and the message
    """
    command = "tweet"  # default command
    if " thread" in message.lower():
        command = "thread"
    return command, message


def make_twitter_api_call(url, params=None, bearer_token=None):
    """Generic method for making requests to the Twitter API."""
    headers = {"Authorization": f"Bearer {bearer_token}"}
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raises an HTTPError for 4XX/5XX errors
        return response.json()
    except requests.exceptions.HTTPError as e:
        logging.error(f"Error making Twitter API call: {e.response.status_code} - {e.response.text}")
        return None


def bearer_oauth(bearer_token):
    """Method required by bearer token authentication."""

    def bearer_auth(request):
        request.headers["Authorization"] = f"Bearer {bearer_token}"
        request.headers["User-Agent"] = "v2UserMentionsPython"
        return request

    return bearer_auth

def connect_to_endpoint(url, params, bearer_token):
    """Connect to Twitter API endpoint."""
    response = requests.get(url, auth=bearer_oauth(bearer_token), params=params)
    logging.info(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()
