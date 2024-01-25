import logging
import os
import time
from datetime import datetime, timedelta

import requests
import tweepy
from src.Llama_index_sandbox.main import initialise_chatbot
from src.Llama_index_sandbox.prompts import TWITTER_THREAD_INPUT
from src.Llama_index_sandbox.retrieve import ask_questions
from src.Llama_index_sandbox.custom_react_agent.tools.reranker.custom_query_engine import CustomQueryEngine
from src.Llama_index_sandbox.utils.gcs_utils import set_secrets_from_cloud
from dotenv import load_dotenv
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
        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(self.auth)

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
        if user_id not in self.last_reply_times:
            return True
        time_since_last_reply = datetime.now() - self.last_reply_times[user_id]
        return time_since_last_reply > timedelta(seconds=30)  # Change the time limit as needed

    def process_webhook_data(self, data, test=False):
        """
        Processes incoming data from the webhook.
        :param data: The data received from the webhook
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
                            message = self.fetch_thread(tweet_id, test=test)
                        elif command == "tweet":
                            message = self.fetch_tweet(tweet_id, test=test)
                        else:
                            message = tweet_text  # Default behavior

                        chat_input = TWITTER_THREAD_INPUT.format(user_input=tweet_text, twitter_thread=message)

                        # Process the message
                        response = self.process_chat_message(chat_input)
                        if response:
                            self.reply_to_tweet(user_id, response, tweet_id)
                            self.last_reply_times[user_id] = datetime.now()
                        else:
                            logging.error("No response generated for the tweet.")
                else:
                    logging.info(f"Rate limit: Not replying to {user_id}")
        else:
            logging.error("Webhook data does not contain tweet creation events.")

    def reply_to_tweet(self, user_id, response, tweet_id):
        """
        Posts a reply to a tweet.
        :param user_id: The user ID to whom the reply should be addressed
        :param response: The response message to be posted
        :param tweet_id: The ID of the tweet being replied to
        """
        try:
            username = self.api.get_user(user_id).screen_name
            reply_text = f"@{username} {response}"
            self.api.update_status(status=reply_text, in_reply_to_status_id=tweet_id)
        except Exception as e:
            logging.error(f"Error posting reply: {e}")

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
            return response
        except Exception as e:
            logging.error(f"Error processing chat message: {e}")
            return None

    import requests

    def fetch_thread(self, tweet_id, test):
        """
        Fetches the parent tweets of the given tweet ID, walking up the conversation tree.
        :param tweet_id: The tweet ID to fetch the thread for
        :return: The fetched thread as a list of tweet texts
        """
        thread = []
        try:
            while tweet_id:
                # Fetch the tweet with the given ID, including note_tweet for long tweets
                tweet_fields = "tweet.fields=created_at,text,referenced_tweets,note_tweet"
                url = f"https://api.twitter.com/2/tweets/{tweet_id}?{tweet_fields}"
                headers = {"Authorization": f"Bearer {self.bearer_token}"}
                response = requests.get(url, headers=headers)
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

    def fetch_tweet(self, tweet_id, test):
        """
        Fetches the tweet with the given tweet ID, attempting to retrieve the full content.
        :param tweet_id: The tweet ID to fetch
        :param test: A test flag to return early with the tweet data for debugging
        :return: The text of the tweet
        """
        try:
            # Fetch the tweet with the given ID, requesting full text and note_tweet for long tweets
            tweet_fields = "tweet.fields=created_at,text,referenced_tweets,note_tweet"
            url = f"https://api.twitter.com/2/tweets/{tweet_id}?{tweet_fields}"
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            response = requests.get(url, headers=headers)
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

    def simulate_webhook_event(self, user_id, tweet_id, tweet_text, command_type="tweet"):
        """
        Simulates a webhook event for testing purposes.

        :param user_id: The user ID of the tweet author
        :param tweet_id: The tweet ID
        :param tweet_text: The text of the tweet
        :param command_type: The command type ('tweet' or 'thread')
        """
        print(f"Simulating webhook event for tweet: {tweet_id}, user: {user_id}, command: {command_type}")
        if command_type == "thread":
            message = self.fetch_thread(tweet_id)
        else:
            message = tweet_text

        # Process the message
        response = self.process_chat_message(message)
        if response:
            print(f"Response: {response}")
        else:
            print("No response generated for the simulated event.")