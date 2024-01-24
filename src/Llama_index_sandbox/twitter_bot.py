import logging
import os
import time
import tweepy
from src.Llama_index_sandbox.main import initialise_chatbot
from src.Llama_index_sandbox.retrieve import ask_questions
from src.Llama_index_sandbox.custom_react_agent.tools.reranker.custom_query_engine import CustomQueryEngine
from src.Llama_index_sandbox.utils.gcs_utils import set_secrets_from_cloud


class TwitterBot:
    """
    A class to represent a Twitter bot that interacts with a chatbot engine.
    """

    def __init__(self):
        """
        Initializes the Twitter Bot with necessary credentials and configurations.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        set_secrets_from_cloud()

        self.consumer_key = os.environ.get('TWITTER_CONSUMER_KEY')
        self.consumer_secret = os.environ.get('TWITTER_CONSUMER_SECRET')
        self.access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')

        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(self.auth)

        self.engine = 'chat'
        self.query_engine_as_tool = True
        self.recreate_index = False
        self.retrieval_engine, self.query_engine, self.store_response_partial, self.config = initialise_chatbot(
            engine=self.engine, query_engine_as_tool=self.query_engine_as_tool, recreate_index=self.recreate_index
        )
        CustomQueryEngine.load_or_compute_weights(document_weight_mappings=CustomQueryEngine.document_weight_mappings,
                                                  weights_file=CustomQueryEngine.weights_file,
                                                  authors_list=CustomQueryEngine.authors_list,
                                                  authors_weights=CustomQueryEngine.authors_weights,
                                                  recompute_weights=True)

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

    @staticmethod
    def retrieve_last_seen_id(file_name):
        """
        Retrieves the ID of the last seen tweet from a file.

        :param file_name: The file from which to read the last seen tweet ID
        :return: The last seen tweet ID
        """
        try:
            with open(file_name, 'r') as f_read:
                last_seen_id = int(f_read.read().strip())
            return last_seen_id
        except FileNotFoundError:
            logging.error("File not found. Starting from the beginning.")
            return 1

    @staticmethod
    def store_last_seen_id(last_seen_id, file_name):
        """
        Stores the ID of the last seen tweet in a file.

        :param last_seen_id: The ID of the last seen tweet
        :param file_name: The file in which to store the ID
        """
        with open(file_name, 'w') as f_write:
            f_write.write(str(last_seen_id))

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

    def reply_to_tweets(self):
        """
        Replies to tweets that mention the bot. Determines whether to process a single tweet or an entire thread.
        """
        last_seen_id = self.retrieve_last_seen_id('last_seen_id.txt')
        mentions = self.api.mentions_timeline(last_seen_id, tweet_mode='extended')

        for mention in reversed(mentions):
            logging.info(f"Replying to {mention.user.screen_name} - {mention.id}")
            last_seen_id = mention.id
            self.store_last_seen_id(last_seen_id, 'last_seen_id.txt')

            command, user_message = self.extract_command_and_message(mention.full_text)

            if command == "thread":
                message = self.fetch_thread(mention.id)
            elif 'quoted_status' in mention._json:
                message = mention.quoted_status.full_text
            else:
                message = user_message

            response = self.process_chat_message(message)
            if response:
                reply = response
                self.api.update_status('@' + mention.user.screen_name + ' ' + reply, mention.id)
            else:
                logging.error("Error processing message or no response generated.")


def main():
    """
    Main function to initialize and run the Twitter Bot.
    """
    bot = TwitterBot()
    while True:
        bot.reply_to_tweets()
        time.sleep(15)  # Sleep to avoid hitting rate limits


if __name__ == "__main__":
    main()
