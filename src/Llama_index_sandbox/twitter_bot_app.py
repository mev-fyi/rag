import logging
import os
import threading
import time
import requests
from flask import Flask
from src.Llama_index_sandbox.twitter_bot import TwitterBot

# Define the Flask app
app = Flask(__name__)

# Configure logging to file and stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("twitter_bot.log"),
        logging.StreamHandler()
    ]
)

# Initialize the TwitterBot
bot = TwitterBot()


# Function to poll Twitter mentions
def poll_twitter_mentions():
    last_polled_id = None
    while True:
        try:
            # Twitter API URL for fetching mentions
            url = 'https://api.twitter.com/1.1/statuses/mentions_timeline.json'
            params = {'count': 20}
            if last_polled_id:
                params['since_id'] = last_polled_id

            # Prepare the authorization header
            headers = {'Authorization': f'Bearer {os.environ["TWITTER_BEARER_TOKEN"]}'}

            # Log the polling attempt
            logging.info("Polling Twitter mentions...")

            # Make the request to the Twitter API
            response = requests.get(url, headers=headers, params=params)

            # Check response status
            if response.status_code == 200:
                mentions = response.json()
                logging.info(f"Fetched {len(mentions)} mentions")

                # Update last_polled_id with the newest mention id
                if mentions:
                    last_polled_id = max(mention['id_str'] for mention in mentions)

                    # Process each mention
                    for mention in mentions:
                        bot.process_mention(mention)
            else:
                logging.error(f"Error fetching mentions: {response.status_code} - {response.text}")

            # Wait before polling again
            time.sleep(12)
        except Exception as e:
            # Log any exception that occurs
            logging.exception("Exception in poll_twitter_mentions")
            # Wait a bit longer before retrying
            time.sleep(60)


# Flask route for a simple health check
@app.route('/')
def hello_world():
    return 'Twitter Bot is running'


# The main entry point for running the Flask app
if __name__ == '__main__':
    # Start the polling thread
    polling_thread = threading.Thread(target=poll_twitter_mentions, daemon=True)
    polling_thread.start()

    # Run the Flask app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
