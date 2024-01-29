import logging
import os
import re

import backoff
import requests


def upload_media_chunked(media_path, auth):
    # INIT request
    init_url = 'https://upload.twitter.com/1.1/media/upload.json'
    files = {
        'command': (None, 'INIT'),
        'media_type': (None, 'image/png'),  # or 'image/jpeg', 'image/gif', 'video/mp4' etc.
        'total_bytes': (None, str(os.path.getsize(media_path))),
    }
    response = requests.post(init_url, files=files, auth=auth)
    media_id = response.json().get('media_id_string')

    # APPEND request
    append_url = 'https://upload.twitter.com/1.1/media/upload.json'
    files = {
        'command': (None, 'APPEND'),
        'media_id': (None, media_id),
        'segment_index': (None, '0'),
        'media': open(media_path, 'rb')
    }
    response = requests.post(append_url, files=files, auth=auth)

    # FINALIZE request
    finalize_url = 'https://upload.twitter.com/1.1/media/upload.json'
    data = {
        'command': 'FINALIZE',
        'media_id': media_id
    }
    response = requests.post(finalize_url, data=data, auth=auth)

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
