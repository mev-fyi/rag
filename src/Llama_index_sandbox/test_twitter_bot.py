import unittest
from unittest.mock import Mock, patch
import twitter_bot_app
from src.Llama_index_sandbox.twitter_bot import TwitterBot


class TestTwitterBot(unittest.TestCase):
    def setUp(self):
        # Initialize the TwitterBot instance
        self.bot = TwitterBot()
        self.bot.api = Mock()  # Mock the Tweepy API

    @patch('twitter_bot_app.twitter_bot.requests.get')
    def test_process_mention_tweet(self, mock_get):
        # Test processing a tweet mention
        tweet_id = '1750323891061633300'
        mention = {'user': {'id_str': '123456'}, 'id_str': tweet_id, 'text': "Explain tweet"}
        mock_get.return_value.json.return_value = {'data': {'text': """The role of single slot finality (SSF) in post-Merge PoS improvement is solidifying. It's becoming clear that SSF is the easiest path to resolving a lot of the Ethereum PoS design's current weaknesses.
            See:
            https://t.co/l8OcVsXCOW
            https://t.co/8sM6DieMjg https://t.co/i6Gz36huW4"""}}
        self.bot.process_mention(mention, test=True)

        # Assertions
        self.bot.api.update_status.assert_called_once()
        call_args = self.bot.api.update_status.call_args[1]
        self.assertIn('@123456', call_args['status'])

    @patch('twitter_bot_app.twitter_bot.requests.get')
    def test_process_mention_thread(self, mock_get):
        # Test processing a thread mention
        thread_tweet_id = '1750534894319497324'
        mention = {'user': {'id_str': '123456'}, 'id_str': thread_tweet_id, 'text': "Explain thread"}
        mock_get.return_value.json.side_effect = [{'data': {'text': 'First tweet of thread'}}, {'data': {'text': 'Second tweet of thread'}}]
        self.bot.process_mention(mention, test=True)

        # Assertions
        self.bot.api.update_status.assert_called_once()
        call_args = self.bot.api.update_status.call_args[1]
        self.assertIn('@123456', call_args['status'])
        self.assertIn('First tweet of thread', call_args['status'])

    def test_fetch_tweet_real_http_request(self):
        # Test fetching a tweet with a real HTTP request
        tweet_id = '1750323891061633300'  # Replace with a valid tweet ID for a real test
        mention = {'user': {'id_str': 'real_user_id'}, 'id_str': tweet_id, 'text': 'Explain this tweet'}
        self.bot.process_mention(mention, test=True, test_http_request=False)
        # Assertions can be added based on the expected outcome

    def test_fetch_tweet_without_http_request(self):
        # Test processing a mention without making an HTTP request
        mention = {'user': {'id_str': '123'}, 'id_str': 'tweet_id_123', 'text': "@bot Explain this tweet"}
        self.bot.process_mention(mention, test=True, test_http_request=True)
        self.bot.api.update_status.assert_called_once()
        call_args = self.bot.api.update_status.call_args[1]
        self.assertIn('@123', call_args['status'])
        self.assertIn('vitalik_ethereum_roadmap_2023', call_args['status'])


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
