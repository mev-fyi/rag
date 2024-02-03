
print("Starting the Twitter bot application...")
import logging
import os
import time
import requests
import dotenv
from concurrent.futures import ThreadPoolExecutor

# Adjust the import statements to match the directory structure in the Docker container
from src.Llama_index_sandbox.twitter_utils import lookup_user_by_username

dotenv.load_dotenv()

from src.Llama_index_sandbox.twitter_bot import TwitterBot
print("Logging configuration initialized...")
# Configure logging to file and stdout
# Determine if running in Docker
is_in_docker = os.getenv('IS_IN_DOCKER', 'false') == 'true'

log_file_path = "/app/twitter_bot.log" if is_in_docker else "twitter_bot.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

# Add initial log to indicate the script has started
logging.info("Starting the Twitter bot application...")

# Initialize the TwitterBot
bot = TwitterBot()

# Fetch the number of workers from an environment variable, with a default fallback
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 5))


# Function to process a single mention
def process_mention(mention):
    try:
        bot.process_mention(mention)
    except Exception as e:
        logging.exception("Exception occurred while processing mention.")


# Function to poll Twitter mentions
def poll_twitter_mentions():
    print("polling twitter mentions")
    last_polled_id = None
    user_id = lookup_user_by_username(os.getenv('TWITTER_USERNAME'))
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    base_url = f'https://api.twitter.com/2/users/{user_id}/mentions'

    logging.info("Twitter bot is ready to poll mentions.")

    while True:
        try:
            params = {
                'max_results': '10',
                'tweet.fields': 'created_at,author_id',
            }
            if last_polled_id:
                params['since_id'] = last_polled_id

            headers = {
                'Authorization': f'Bearer {bearer_token}',
                'User-Agent': 'v2MentionsPython'
            }

            logging.info("Polling for mentions...")
            response = requests.get(base_url, headers=headers, params=params)

            if response.status_code == 200:
                response_json = response.json()
                mentions = response_json.get('data', [])
                logging.info(f"Received {len(mentions)} mentions.")

                # Process mentions using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    for mention in mentions:
                        executor.submit(process_mention, mention)

                meta_data = response_json.get('meta', {})
                if 'newest_id' in meta_data:
                    last_polled_id = meta_data['newest_id']
            else:
                logging.error(f"Error fetching mentions: {response.status_code} - {response.text}")

            time.sleep(13)
        except Exception as e:
            logging.exception("Exception occurred while polling mentions.")
            time.sleep(120)


if __name__ == '__main__':
    print("starting main")
    poll_twitter_mentions()
