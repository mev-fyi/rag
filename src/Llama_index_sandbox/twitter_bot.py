import logging
import os
import time
import tweepy
from src.Llama_index_sandbox.main import initialise_chatbot
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

    def process_webhook_data(self, data):
        """
        Processes incoming data from the webhook.
        :param data: The data received from the webhook
        """
        # Extract relevant information from the data
        if 'tweet_create_events' in data:
            for event in data['tweet_create_events']:
                user_id = event['user']['id_str']
                tweet_id = event['id_str']
                tweet_text = event['text']

                # Check if the tweet is a reply or quote
                if 'in_reply_to_status_id_str' in event or 'quoted_status' in event:
                    command, _ = self.extract_command_and_message(tweet_text)

                    if command == "thread":
                        message = self.fetch_thread(tweet_id)
                    elif command == "tweet":
                        message = tweet_text
                    else:
                        message = tweet_text  # Default behavior

                    # Process the message
                    response = self.process_chat_message(message)
                    if response:
                        self.reply_to_tweet(user_id, response, tweet_id)
                    else:
                        logging.error("No response generated for the tweet.")
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

    def fetch_thread(self, tweet_id):
        """
        Fetches the entire thread of tweets leading up to the given tweet ID.

        :param tweet_id: The tweet ID from which to start fetching the thread
        :return: The fetched thread as a single concatenated string
        """
        thread = []
        while tweet_id:
            try:
                tweet = self.api.get_status(tweet_id, tweet_mode='extended')
                thread.append(tweet.full_text)
                tweet_id = tweet.in_reply_to_status_id
            except tweepy.TweepError as e:
                logging.error(f"Error fetching tweet: {e}")
                break
        return " ".join(reversed(thread))

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

