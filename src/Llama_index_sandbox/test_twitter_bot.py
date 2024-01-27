import unittest
from unittest.mock import Mock, patch

import os
import sys
# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Llama_index_sandbox import twitter_bot_app  # Make sure this is the correct path to your Flask app module
from src.Llama_index_sandbox.twitter_bot import TwitterBot  # Update the import path according to your project structure


class TestTwitterBotAndApp(unittest.TestCase):
    def setUp(self):
        self.app = twitter_bot_app.app.test_client()
        self.app.testing = True
        self.bot = TwitterBot()  # Using real TwitterBot instance

    def test_hello_world_route(self):
        # Test the '/' route of Flask app
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Twitter Bot is running', response.get_data(as_text=True))

    def test_process_mention_thread(self):
        # Test processing a thread mention with real TwitterBot instance
        thread_tweet_id = '1750534894319497324'
        mention = {'user': {'id_str': '123456'}, 'id_str': thread_tweet_id, 'text': "Explain thread"}
        self.bot.process_mention(mention, test=True, test_http_request=False, post_reply_in_prod=True, is_paid_account=False)

        # Since this is a real API call, assertions will depend on the actual response
        # Ensure you have appropriate checks to verify the bot's behavior

    # Additional tests for different scenarios of your TwitterBot can be added here
    # Remember to respect Twitter's rate limits and usage policies


class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        self.app = twitter_bot_app.app.test_client()
        self.app.testing = True

    def test_hello_world_route(self):
        # Test the '/' route
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Twitter Bot is running', response.get_data(as_text=True))


if __name__ == '__main__':
    unittest.main()
