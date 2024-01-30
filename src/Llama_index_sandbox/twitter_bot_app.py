import logging
import os
import threading
import time
import requests
from flask import Flask
import dotenv
dotenv.load_dotenv()

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
def poll_twitter_mentions_v1():
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


def poll_twitter_mentions():
    last_polled_id = None
    # Replace with your user ID
    user_id = lookup_user_by_username(os.getenv('TWITTER_USERNAME'))
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

    # Set the base URL for the new Twitter API v2 endpoint
    base_url = f'https://api.twitter.com/2/users/{user_id}/mentions'

    while True:
        try:
            # Prepare the request parameters and headers
            params = {
                'max_results': '10',
                'tweet.fields': 'created_at,author_id',
                # Add other parameters as required by your application
            }
            if last_polled_id:
                params['since_id'] = last_polled_id

            headers = {
                'Authorization': f'Bearer {bearer_token}',
                'User-Agent': 'v2MentionsPython'
            }

            # Log the attempt to poll for mentions
            logging.info("Polling for mentions...")

            # Make the GET request to the Twitter API
            response = requests.get(base_url, headers=headers, params=params)

            # Check the response status code
            if response.status_code == 200:
                response_json = response.json()
                mentions = response_json.get('data', [])
                logging.info(f"Received {len(mentions)} mentions.")

                # Handle and process each mention
                for mention in mentions:
                    # Process the mention here
                    bot.process_mention(mention)

                # Update the last_polled_id if new mentions were fetched
                meta_data = response_json.get('meta', {})
                if 'newest_id' in meta_data:
                    last_polled_id = meta_data['newest_id']
            else:
                logging.error(f"Error fetching mentions: {response.status_code} - {response.text}")

            # Sleep before the next polling attempt
            time.sleep(60)
        except Exception as e:
            # Log the exception and wait before retrying
            logging.exception("Exception occurred while polling mentions.")
            time.sleep(120)

# Flask route for a simple health check
@app.route('/')
def hello_world():
    return 'Twitter Bot is running'


# The main entry point for running the Flask app
if __name__ == '__main__':
    # Start the polling thread
    # polling_thread = threading.Thread(target=poll_twitter_mentions, daemon=True)
    # polling_thread.start()
    # Instead of starting a separate thread, run polling in the main thread
    # This will block the execution, so the server will not respond to HTTP requests
    # TODO 2024-01-30: TBD to deploy on continuously running Cloud Run or GKE.
    #  Cloud Run makes lenss sense since the app is polling all mentions anyway but processes them serially. Need some more architecture thoughts
    poll_twitter_mentions()

    # Run the Flask app
    # port = int(os.environ.get('PORT', 8080))
    # app.run(host='0.0.0.0', port=port)
