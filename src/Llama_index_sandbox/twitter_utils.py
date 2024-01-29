import os
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
