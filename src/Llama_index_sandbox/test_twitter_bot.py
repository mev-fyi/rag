import unittest
from unittest.mock import Mock, patch

import os
import sys

from src.Llama_index_sandbox.twitter_bot_app import poll_twitter_mentions
from src.Llama_index_sandbox.twitter_utils import lookup_user_by_username

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Llama_index_sandbox import twitter_bot_app  # Make sure this is the correct path to your Flask app module
from src.Llama_index_sandbox.twitter_bot import TwitterBot  # Update the import path according to your project structure


class TestTwitterBotAndApp(unittest.TestCase):
    def setUp(self):
        pass
        self.bot = TwitterBot()  # Using real TwitterBot instance

    def test_process_mention_thread(self):
        # Test processing a thread mention with real TwitterBot instance
        thread_tweet_id = '1756724189992456354'
        # author_id = '967029293124669442'
        mention = {'author_id': lookup_user_by_username('uriklarman'), 'id': thread_tweet_id, 'text': "Explain thread"}
        # NOTE: set test_http_request to True to test the method while doing GET calls to Twitter API to fecth the thread back
        self.bot.process_mention(mention, test=True, test_http_request=True, post_reply_in_prod=False, is_paid_account=False)

    # def test_poll_twitter_mentions(self):
    #     # Test processing a thread mention with real TwitterBot instance
    #     poll_twitter_mentions()


    # Additional tests for different scenarios of your TwitterBot can be added here
    # Remember to respect Twitter's rate limits and usage policies

    #def test_create_shared_chat(self):
    #    bot = TwitterBot()
    #
    #    # Define a mock chat response and metadata
    #    chat_response = Mock()
    #    chat_response.response = "This is a test response for the chatbot"
    #    metadata = "Test metadata"
    #
    #    # Call the method under test
    #    shared_chat_link = bot.create_shared_chat(chat_response, metadata)
    #
    #    # Test if the shared chat link is returned as expected
    #    self.assertIsNotNone(shared_chat_link)



if __name__ == '__main__':
    unittest.main()
