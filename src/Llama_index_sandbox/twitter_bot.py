import logging
import os
import time
from datetime import datetime, timedelta

import requests
from requests_oauthlib import OAuth1
import tweepy
from tweepy import OAuth1UserHandler

from src.Llama_index_sandbox import globals as glb
from src.Llama_index_sandbox import constants
from src.Llama_index_sandbox.main import initialise_chatbot
from src.Llama_index_sandbox.prompts import TWITTER_THREAD_INPUT
from src.Llama_index_sandbox.retrieve import ask_questions
from src.Llama_index_sandbox.custom_react_agent.tools.reranker.custom_query_engine import CustomQueryEngine
from src.Llama_index_sandbox.twitter_utils import safe_request, split_response_into_tweets, TWEET_CHAR_LENGTH, vitalik_ethereum_roadmap_2023, take_screenshot_and_upload
from src.Llama_index_sandbox.utils.gcs_utils import set_secrets_from_cloud
from dotenv import load_dotenv

from src.Llama_index_sandbox.utils.utils import get_last_index_embedding_params

load_dotenv()


class TwitterBot:
    """
    A class to represent a Twitter bot that interacts with a chatbot engine,
    designed to be triggered via webhooks.
    """

    def __init__(self):
        """
        Initializes the Twitter Bot with necessary credentials and configurations.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        set_secrets_from_cloud()

        # Twitter API Authentication
        self.consumer_key = os.environ.get('TWITTER_CONSUMER_KEY')
        self.consumer_secret = os.environ.get('TWITTER_CONSUMER_SECRET')
        self.access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')
        self.bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
        self.auth = OAuth1UserHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(self.auth)

        # GCS screenshot store
        self.gcs_bucket = os.environ.get('GCS_BUCKET')

        # Chatbot Engine Initialization
        self.engine = 'chat'
        self.query_engine_as_tool = True
        self.recreate_index = False
        self.retrieval_engine, self.query_engine, self.store_response_partial, self.config = initialise_chatbot(
            engine=self.engine, query_engine_as_tool=self.query_engine_as_tool, recreate_index=self.recreate_index
        )
        CustomQueryEngine.load_or_compute_weights(
            document_weight_mappings=CustomQueryEngine.document_weight_mappings,
            weights_file=CustomQueryEngine.weights_file,
            authors_list=CustomQueryEngine.authors_list,
            authors_weights=CustomQueryEngine.authors_weights,
            recompute_weights=True
        )
        self.last_reply_times = {}

    def should_reply_to_user(self, user_id):
        """
        Determines if the bot should reply to a specific user based on rate limiting.
        :param user_id: The user ID to check
        :return: True if the bot should reply, False otherwise
        """
        # Check if the user_id is the bot's own user_id to avoid self-reply loop
        my_user_id = self.api.verify_credentials().id_str
        if user_id == my_user_id:
            return False
        if user_id not in self.last_reply_times:
            return True
        time_since_last_reply = datetime.now() - self.last_reply_times[user_id]
        return time_since_last_reply > timedelta(seconds=30)  # Change the time limit

    def process_webhook_data(self, data, test=False, test_http_request=False):
        """
        Processes incoming data from the webhook.
        :param data: The data received from the webhook

        Args:
            test:
            test_http_request:
        """
        # Extract relevant information from the data
        if 'tweet_create_events' in data:
            for event in data['tweet_create_events']:
                user_id = event['user']['id_str']
                if self.should_reply_to_user(user_id):
                    tweet_id = event['id_str']
                    tweet_text = event['text']

                    # Check if the tweet is a reply or quote
                    if 'in_reply_to_status_id_str' in event or 'quoted_status' in event:
                        command, _ = self.extract_command_and_message(tweet_text)

                        if command == "thread":
                            message = self.fetch_thread(tweet_id, test=test, test_http_request=test_http_request)
                        elif command == "tweet":
                            message = self.fetch_tweet(tweet_id, test=test, test_http_request=test_http_request)
                        else:
                            message = tweet_text  # Default behavior

                        if message is None:
                            logging.error("Could not fetch tweet")
                            return

                        chat_input = TWITTER_THREAD_INPUT.format(user_input=tweet_text, twitter_thread=message)
                        # TODO 2024-01-25: if the thread or tweet is referring to document existing in the database, fetch their content too.
                        # TODO 2024-01-25: if there is one or more images to each tweet, add them.

                        # Process the message
                        response = self.process_chat_message(chat_input).response
                        if response:
                            self.reply_to_tweet(user_id, response, tweet_id, test)
                            self.last_reply_times[user_id] = datetime.now()
                        else:
                            logging.error("No response generated for the tweet.")
                else:
                    logging.info(f"Rate limit: Not replying to {user_id}")
        else:
            logging.error("Webhook data does not contain tweet creation events.")

    def fetch_username_directly(self, user_id):
        """
        Fetches the username of a user directly using the Twitter API v2.
        :param user_id: The user ID
        :return: The username of the user
        """
        url = f"https://api.twitter.com/2/users/{user_id}"
        headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                user_data = response.json().get('data', {})
                return user_data.get('username')
            else:
                logging.error(f"Error fetching username directly: {response.status_code} - {response.text}")
        except Exception as e:
            logging.error(f"Exception in fetch_username_directly: {e}")
        return None

    def reply_to_tweet(self, user_id, response, tweet_id, test, command, media_id=None, post_reply_in_prod=True, is_paid_account=False):
        """
        Posts a reply to a tweet. If the account is not paid, splits the response into multiple tweets.
        :param user_id: The user ID to whom the reply should be addressed
        :param response: The response message to be posted
        :param tweet_id: The ID of the tweet being replied to
        :param test: Boolean flag for testing
        :param command: What the user wants explained: thread or tweet
        :param media_id: Media id once uploaded on Twitter e.g. screenshot from the shared_link
        :param post_reply_in_prod:
        :param is_paid_account: Boolean flag indicating if the account is a paid subscription
        """
        if post_reply_in_prod:
            try:
                username = self.fetch_username_directly(user_id) if not test else 'unlock_VALue'
                if username:
                    reply_text = f"@{username} Here is your {command} explanation: {response}"
                    if is_paid_account or len(reply_text) <= TWEET_CHAR_LENGTH:
                        self.direct_reply_to_tweet(tweet_id, reply_text, tweet_number=0, media_id=media_id)
                    else:
                        self.post_thread_reply(username, response, tweet_id)
            except Exception as e:
                logging.warning(f"Error posting reply with direct posting, now retrying with Tweepy: {e}")
                # Fallback to fetch username directly and then reply
                username = self.api.get_user(user_id=user_id).screen_name if not test else 'unlock_VALue'
                reply_text = f"@{username} {response}"
                self.api.update_status(status=reply_text, in_reply_to_status_id=tweet_id)
        else:
            logging.info("Not posting replies in PROD")

    def post_thread_reply(self, username, response, tweet_id):
        """
        Posts a reply as a thread of tweets.
        :param username: The username to whom the reply should be addressed
        :param response: The response message to be posted
        :param tweet_id: The ID of the tweet being replied to
        """
        tweet_chunks = split_response_into_tweets(response, username)
        previous_tweet_id = None

        for i, chunk in enumerate(tweet_chunks):
            logging.info(f"Tweet chunk {i}: {chunk}")

            if i == 0:
                reply_text = f"@{username} {chunk}"
                previous_tweet_id = self.direct_reply_to_tweet(tweet_id, reply_text, tweet_number=i)
            else:
                reply_text = chunk
                previous_tweet_id = self.direct_reply_to_tweet(tweet_id, reply_text, tweet_number=i, in_thread=True, previous_tweet_id=previous_tweet_id)
            time.sleep(1)

        logging.info("Completed tweeting!")

    def direct_reply_to_tweet(self, tweet_id, reply_text, tweet_number, media_id=None, in_thread=False, previous_tweet_id=None):
        """
        Posts a reply using a direct Twitter API call with OAuth 1.0a.
        Can also be used to post a thread by linking tweets.
        :param tweet_id: The ID of the tweet being replied to
        :param reply_text: The reply message to be posted
        :param tweet_number: Number of the tweet in a sequence
        :param media_id: ID of the media to be attached (optional)
        :param in_thread: Boolean flag indicating if this is part of a thread
        :param previous_tweet_id: The ID of the previous tweet in the thread (if applicable)
        :return: The ID of the new tweet
        """
        url = 'https://api.twitter.com/2/tweets'
        reply_to_id = previous_tweet_id if in_thread and previous_tweet_id else tweet_id

        # Payload for the tweet
        payload = {
            "text": reply_text,
            "reply": {
                "in_reply_to_tweet_id": reply_to_id
            }
        }

        # Add media_id if available
        if media_id:
            payload['attachments'] = {
                'media_keys': [f"media_key:{media_id}"]
            }

        # Create an OAuth1 object
        auth = OAuth1(
            client_key=self.consumer_key,
            client_secret=self.consumer_secret,
            resource_owner_key=self.access_token,
            resource_owner_secret=self.access_token_secret
        )

        headers = {
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, headers=headers, json=payload, auth=auth)
            if response.status_code == 201:
                logging.info(f"Reply posted tweet # [{tweet_number}] successfully with direct API call.")
                new_tweet_id = response.json()['data']['id']
                return new_tweet_id
            else:
                logging.error(f"Error posting reply with direct API call: {response.status_code} - {response.text}")
        except Exception as e:
            logging.error(f"Error with direct API call: {e}")

        return None

    def process_chat_message(self, message):
        """
        Processes a chat message using the chatbot engine and returns the response.

        :param message: The message to process
        :return: The response from the chatbot
        """
        try:
            response, formatted_metadata = ask_questions(
                input_queries=[message],
                retrieval_engine=self.retrieval_engine,
                query_engine=self.query_engine,
                store_response_partial=self.store_response_partial,
                engine=self.engine,
                query_engine_as_tool=self.query_engine_as_tool,
                chat_history=[],
                run_application=True,
                reset_chat=self.config.reset_chat
            )
            return response, formatted_metadata
        except Exception as e:
            logging.error(f"Error processing chat message: {e}")
            return None, None

    def fetch_thread(self, tweet_id, test, test_http_request):
        """
        Fetches the parent tweets of the given tweet ID, walking up the conversation tree.
        :param tweet_id: The tweet ID to fetch the thread for
        :return: The fetched thread as a list of tweet texts
        """
        if test and not test_http_request:
            return vitalik_ethereum_roadmap_2023
        thread = []
        try:
            while tweet_id:
                # Fetch the tweet with the given ID, including note_tweet for long tweets
                tweet_fields = "tweet.fields=created_at,text,referenced_tweets,note_tweet"
                url = f"https://api.twitter.com/2/tweets/{tweet_id}?{tweet_fields}"
                headers = {"Authorization": f"Bearer {self.bearer_token}"}
                response = safe_request(url, headers)
                if not response:
                    logging.error(f"Failed to fetch tweet with ID: {tweet_id}")
                    return None
                response.raise_for_status()
                tweet_data = response.json().get('data', {})
                tweet_text = tweet_data.get('note_tweet', {}).get('text') or tweet_data.get('text')
                thread.append(tweet_text)

                # Check if this tweet is in reply to another and get the parent tweet ID
                referenced_tweets = tweet_data.get('referenced_tweets', [])
                parent_tweet = next((ref for ref in referenced_tweets if ref['type'] == 'replied_to'), None)
                tweet_id = parent_tweet['id'] if parent_tweet else None

        except requests.exceptions.HTTPError as e:
            logging.error(f"Error fetching tweet: {e.response.text}")
            return None

        thread = thread[::-1]  # Reverse to maintain the chronological order
        return '\n'.join(thread)

    # Additional method within the TwitterBot class

    def fetch_tweet(self, tweet_id, test, test_http_request):
        """
        Fetches the tweet with the given tweet ID, attempting to retrieve the full content.
        :param tweet_id: The tweet ID to fetch
        :param test: A test flag to return early with the tweet data for debugging
        :return: The text of the tweet
        """
        if test and not test_http_request:
            vitalik_tweet = """
            The role of single slot finality (SSF) in post-Merge PoS improvement is solidifying. It's becoming clear that SSF is the easiest path to resolving a lot of the Ethereum PoS design's current weaknesses.
            See:
            https://t.co/l8OcVsXCOW
            https://t.co/8sM6DieMjg https://t.co/i6Gz36huW4
            """
            return vitalik_tweet
        try:
            # Fetch the tweet with the given ID, requesting full text and note_tweet for long tweets
            tweet_fields = "tweet.fields=created_at,text,referenced_tweets,note_tweet"
            url = f"https://api.twitter.com/2/tweets/{tweet_id}?{tweet_fields}"
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            response = safe_request(url, headers)
            if not response:
                logging.error(f"Failed to fetch tweet with ID: {tweet_id}")
                return None
            response.raise_for_status()
            tweet_data = response.json().get('data', {})

            # Use the note_tweet text if available, otherwise use the standard text
            tweet_text = tweet_data.get('note_tweet', {}).get('text') or tweet_data.get('text')

            if test:
                return tweet_text

            # If the tweet is in reply to or quoting another tweet, fetch its details
            referenced_tweets = tweet_data.get('referenced_tweets', [])
            parent_tweet = next((ref for ref in referenced_tweets if ref['type'] in ['replied_to', 'quoted']), None)
            if parent_tweet:
                parent_tweet_id = parent_tweet['id']
                parent_tweet_url = f"https://api.twitter.com/2/tweets/{parent_tweet_id}?{tweet_fields}"
                parent_response = requests.get(parent_tweet_url, headers=headers)
                parent_response.raise_for_status()
                parent_tweet_data = parent_response.json().get('data', {})
                return parent_tweet_data.get('note_tweet', {}).get('text') or parent_tweet_data.get('text')

        except requests.exceptions.HTTPError as e:
            logging.error(f"Error fetching tweet: {e.response.text}")
            return None

        return None  # Return None if it's an original tweet or in case of an error

    def process_mention(self, mention, test=False, test_http_request=False, post_reply_in_prod=True, is_paid_account=False):
        """
        Processes a single mention from the Twitter mentions timeline.
        :param mention: The tweet data (mention) to be processed
        :param test: Flag for testing
        :param is_paid_account: Flag to indicate if it's a paid account

        Args:
            test_http_request:
        """
        user_id = mention['author_id']
        tweet_id = mention['id']

        if tweet_id in self.last_reply_times:
            first_reply_id = self.last_reply_times[user_id]
            already_processed_reply = f"Already processed this mention. See first reply: {first_reply_id}"
            self.api.update_status(status=already_processed_reply, in_reply_to_status_id=tweet_id)
            logging.info(f"Mention already processed: {tweet_id}")
            return

        tweet_text = mention['text']

        if self.should_reply_to_user(user_id):
            # Check if the mention is a reply or a quote
            command, _ = self.extract_command_and_message(tweet_text)

            if command == "thread":
                message = self.fetch_thread(tweet_id, test=test, test_http_request=test_http_request)
            elif command == "tweet":
                message = self.fetch_tweet(tweet_id, test=test, test_http_request=test_http_request)
            else:
                message = tweet_text  # Default behavior for standalone mentions

            if message is None:
                logging.error("Could not fetch tweet")
                return

            chat_input = TWITTER_THREAD_INPUT.format(user_input=tweet_text, twitter_thread=message)
            # Process the message
            chat_response, metadata = self.process_chat_message(chat_input)
            if chat_response:
                shared_chat_link = self.create_shared_chat(chat_response, metadata)
                media_id = take_screenshot_and_upload(url=shared_chat_link)
                self.reply_to_tweet(user_id=user_id, response=shared_chat_link, tweet_id=tweet_id, test=test, command=command, media_id=media_id, post_reply_in_prod=post_reply_in_prod, is_paid_account=is_paid_account)
                self.last_reply_times[user_id] = tweet_id  # Update with the latest processed tweet ID
            else:
                logging.error("No response generated for the mention.")
        else:
            logging.info(f"Rate limit: Not replying to {user_id}")

    def create_shared_chat(self, chat_response, metadata):
        """
        Sends a POST request to the Next.js API to create a shared chat.
        :param messages: The chat messages to be sent
        :return: The shared chat link or None in case of an error
        """
        url = os.environ.get('NEXTJS_API_ENDPOINT')  # Fetch the API endpoint from environment variables
        api_key = os.environ.get('NEXTJS_API_KEY')  # Fetch the API key from environment variables

        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key
        }
        embedding_model_name, text_splitter_chunk_size, chunk_overlap, _ = get_last_index_embedding_params()

        model_specifications = {
            "embedding_model_parameters": {
                "embedding_model_name": embedding_model_name,
                "text_splitter_chunk_size": text_splitter_chunk_size,
                "chunk_overlap": chunk_overlap,
                "number of chunks to retrieve": glb.NUMBER_OF_CHUNKS_TO_RETRIEVE,  # NOTE 2023-10-30: fix the retrieval of this as global variable
                "temperature": constants.LLM_TEMPERATURE,
            }
        }

        data = {
            "status": "completed",
            "response": chat_response.response,
            "formatted_metadata": metadata,
            "job_id": '',
            "model_specifications": model_specifications,
        }

        logging.info(f'POSTing response to front-end: {data}')

        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                shared_chat_link = response.json().get('sharedChatLink')
                logging.info(f'Shared chat created successfully: {shared_chat_link}')
                return shared_chat_link
            else:
                logging.error(f"Error creating shared chat: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logging.error(f"Exception in create_shared_chat: {e}")
            return None

    @staticmethod
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

