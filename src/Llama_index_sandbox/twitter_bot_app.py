import logging
import os
import threading
import time
import requests
from flask import Flask
from src.Llama_index_sandbox.twitter_bot import TwitterBot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
bot = TwitterBot()


def poll_twitter_mentions():
    last_polled_id = None
    while True:
        try:
            url = 'https://api.twitter.com/1.1/statuses/mentions_timeline.json'
            params = {'count': 20}
            if last_polled_id:
                params['since_id'] = last_polled_id

            headers = {'Authorization': f'Bearer {os.environ["TWITTER_BEARER_TOKEN"]}'}

            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                mentions = response.json()
                if mentions:
                    # Update last_polled_id to the highest ID from the fetched mentions
                    last_polled_id = max(mention['id_str'] for mention in mentions)
                    for mention in mentions:
                        bot.process_mention(mention)
            else:
                logging.error(f"Error fetching mentions: {response.status_code}")

            time.sleep(12)  # Sleep for 12 seconds before polling again
        except Exception as e:
            logging.exception("Exception in poll_twitter_mentions")
            time.sleep(60)  # Wait a bit longer if there's an error


@app.route('/')
def hello_world():
    return 'Twitter Bot is running'


if __name__ == '__main__':
    polling_thread = threading.Thread(target=poll_twitter_mentions)
    polling_thread.start()

    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
