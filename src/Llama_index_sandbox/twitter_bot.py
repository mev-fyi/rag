import logging
import os
import time
from datetime import datetime, timedelta

import requests
from requests_oauthlib import OAuth1
import tweepy
from tweepy import OAuth1UserHandler

from src.Llama_index_sandbox.main import initialise_chatbot
from src.Llama_index_sandbox.prompts import TWITTER_THREAD_INPUT, TWITTER_TWEET_INPUT
from src.Llama_index_sandbox.retrieve import ask_questions
from src.Llama_index_sandbox.custom_react_agent.tools.reranker.custom_query_engine import CustomQueryEngine
from src.Llama_index_sandbox.twitter_utils import safe_request, split_response_into_tweets, TWEET_CHAR_LENGTH, TWEET_CHAR_LENGTH_FOR_LINE_RETURN, vitalik_ethereum_roadmap_2023, take_screenshot_and_upload, lookup_user_by_username, connect_to_endpoint, extract_command_and_message, create_shared_chat, fetch_username_directly
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
        if (not os.environ.get('ENVIRONMENT', 'LOCAL') == 'LOCAL') or not os.environ.get('ENVIRONMENT') == 'STAGING':
            # NOTE 2024-02-01: staging is when we still want 'LOCAL' logs while deployed on cloud
            set_secrets_from_cloud()

        # Twitter API Authentication
        self.user_id = lookup_user_by_username(os.getenv('TWITTER_USERNAME'))
        self.username = os.getenv('TWITTER_USERNAME')
        self.consumer_key = os.environ.get('TWITTER_CONSUMER_KEY')
        self.consumer_secret = os.environ.get('TWITTER_CONSUMER_SECRET')
        self.access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')
        self.bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
        self.auth = OAuth1UserHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(self.auth)
        self.take_screenshot = os.environ.get('TAKE_SCREENSHOT', 'FALSE') == "TRUE"
        self.seconds_throttler_for_user = os.environ.get('SECONDS_THROTTLER_FOR_USER', 30)

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
        # File to store processed mentions
        self.cached_already_replied_to_tweet_ids, self.cached_invalid_tweets = self.initialize_reply_status_ids()

    def initialize_reply_status_ids(self):
        """
        Initializes a set of tweet IDs the bot has replied to, with pagination to retrieve up to 3200 tweets.
        This simplifies the checking process for whether a mention has already been responded to.
        """
        # We get all the tweets of the bot
        url = f"https://api.twitter.com/2/users/{self.user_id}/tweets"
        # Then using the `expansions` parameter "referenced_tweets.id" we find all the parent tweets of the bot.
        # Once the tag "@mevfyi" shows up, then it means that we already responded to that and we no longer need to process it.
        # Then we loop in the mentions, and if the mention is in the list that we just constituted in the initialize_reply_status_ids method,
        # then we do not need to process it.
        params = {
            "max_results": "100",
            "expansions": "referenced_tweets.id"  # https://developer.twitter.com/en/docs/twitter-api/tweets/timelines/api-reference/get-users-id-tweets
        }

        all_in_reply_to_user_ids = []
        all_invalid_tweets = []
        total_fetched = 0
        max_tweets = 3200  # Maximum number of tweets to fetch
        
        while True:
            response = connect_to_endpoint(url, params, self.bearer_token)
            if response:
                tweets = response.get('includes', [])['tweets']  # check a link to find about expansions and includes https://developer.twitter.com/en/docs/twitter-api/tweets/timelines/api-reference/get-users-id-tweets
                fetched_tweet_ids = []
                invalid_tweets = []
                # First return the list of tweet IDs where the user mentioned the bot
                for tweet in tweets:
                    if self.check_if_valid_mention(mention_text=tweet['text']):
                        fetched_tweet_ids.append(tweet['id'])
                    else:
                        invalid_tweets.append(tweet['id'])

                # 2024-02-03: We simplify the "is this a user mentioning the bot" process by checking in the text if there is the "@mevfyi" tag
                # otherwise I'd have to look up the user corresponding to each of those tweet which is likely to explode in the number of calls, while we have the texts by defaults anyway.

                logging.info(f"[initialize_reply_status_ids] Adding tweet IDs of [{self.username}] mentions: {fetched_tweet_ids} to cached_already_replied_to_tweet_ids and {invalid_tweets} to cached_invalid_tweets")
                # Check if we have fetched the maximum number or if there are no more tweets
                all_in_reply_to_user_ids += fetched_tweet_ids
                all_invalid_tweets += invalid_tweets
                total_fetched += len(fetched_tweet_ids)

                if total_fetched >= max_tweets or 'next_token' not in response.get('meta', {}):
                    break

                # Set the next_token for the next iteration
                params['pagination_token'] = response['meta']['next_token']
            else:
                break  # Exit if there is an error in the response
                
        return all_in_reply_to_user_ids, all_invalid_tweets

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

        # Convert stored timestamp to datetime object
        last_reply_time = datetime.fromtimestamp(self.last_reply_times[user_id])
        time_since_last_reply = datetime.now() - last_reply_time

        return time_since_last_reply > timedelta(seconds=self.seconds_throttler_for_user)

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
        try:
            username = fetch_username_directly(bearer_token=self.bearer_token, user_id=user_id) if not test else 'unlock_VALue'
            if username:
                reply_text = f"@{username} your {command} explanation: {response}"
                reply_text = reply_text.replace(f"@{self.username}", '')  # let's make sure the user can't trick the bot tagging itself

                if is_paid_account or len(reply_text) <= TWEET_CHAR_LENGTH:
                    self.direct_reply_to_tweet(tweet_id, reply_text, tweet_number=0, media_id=media_id, post_reply_in_prod=post_reply_in_prod)
                else:
                    self.post_thread_reply(username, response, tweet_id)
        except Exception as e:
            logging.warning(f"Error posting reply with direct posting, now retrying with Tweepy: {e}")
            # Fallback to fetch username directly and then reply
            username = self.api.get_user(user_id=user_id).screen_name if not test else 'unlock_VALue'
            reply_text = f"@{username} {response}"
            self.api.update_status(status=reply_text, in_reply_to_status_id=tweet_id)

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

    def direct_reply_to_tweet(self, tweet_id, reply_text, tweet_number, media_id=None, in_thread=False, previous_tweet_id=None, post_reply_in_prod=False):
        """
        Posts a reply using a direct Twitter API call with OAuth 1.0a.
        Can also be used to post a thread by linking tweets.
        :param tweet_id: The ID of the tweet being replied to
        :param reply_text: The reply message to be posted
        :param tweet_number: Number of the tweet in a sequence
        :param media_id: ID of the media to be attached (optional)
        :param in_thread: Boolean flag indicating if this is part of a thread
        :param previous_tweet_id: The ID of the previous tweet in the thread (if applicable)
        :param post_reply_in_prod: If we post the bot reply in PROD as in tweet it away
        :return: The ID of the new tweet
        """
        url = 'https://api.twitter.com/2/tweets'
        reply_to_id = previous_tweet_id if in_thread and previous_tweet_id else tweet_id

        # TODO 2024-02-07: investigate https://twitter.com/mevfyi/with_replies diff between post and reply
        # Payload for the tweet
        if reply_to_id is None or not reply_to_id:
            logging.warning(f"[direct_reply_to_tweet] reply_to_id is unspecified: [{reply_to_id}]")
        payload = {
            "text": reply_text,
            "reply": {
                "in_reply_to_tweet_id": reply_to_id
            }
        }

        # Add media_id if available
        if media_id:
            payload['media'] = {
                'media_ids': [media_id]  # media_id should be a string
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
        if post_reply_in_prod:
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
        else:
            logging.info("Not posting reply \n\n```[{reply_text}]```\n\n in prod.")

    def process_chat_message(self, message, direct_llm_call=False):
        """
        Processes a chat message using the chatbot engine and returns the response.

        :param message: The message to process
        :param direct_llm_call: To directly make an LLM call without passing by the agent
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
                reset_chat=self.config.reset_chat,
                direct_llm_call=direct_llm_call
            )
            return response, formatted_metadata
        except Exception as e:
            logging.error(f"Error processing chat message: {e}")
            return None, None

    def fetch_thread(self, tweet_id, test, test_http_request, max_retries=5, backoff_factor=60):
        if test and not test_http_request:
            return vitalik_ethereum_roadmap_2023

        thread = []
        attempt = 0

        while tweet_id and attempt < max_retries:
            try:
                tweet_fields = "tweet.fields=created_at,text,referenced_tweets,note_tweet"
                url = f"https://api.twitter.com/2/tweets/{tweet_id}?{tweet_fields}"
                headers = {"Authorization": f"Bearer {self.bearer_token}"}
                response = connect_to_endpoint(url, None, self.bearer_token, max_retries, backoff_factor)

                tweet_data = response.get('data', {})
                tweet_text = tweet_data.get('note_tweet', {}).get('text') or tweet_data.get('text')
                thread.append(tweet_text)

                referenced_tweets = tweet_data.get('referenced_tweets', [])
                parent_tweet = next((ref for ref in referenced_tweets if ref['type'] == 'replied_to'), None)
                tweet_id = parent_tweet['id'] if parent_tweet else None
                attempt = 0  # Reset attempt counter after successful fetch

            except Exception as e:
                logging.error(f"Error fetching tweet: {e}")
                attempt += 1
                if attempt >= max_retries:
                    break
                sleep_time = backoff_factor * (2 ** attempt)
                logging.info(f"Waiting for {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)

        thread = thread[::-1]  # Reverse to maintain the chronological order
        return '\n'.join(thread)

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

    def check_if_valid_mention(self, mention_text: str):
        # this relies on the fact that the bot doesn't mention itself as per the reply method
        if (f"@{self.username}" in mention_text) and ("explain" in mention_text.lower()):
            return True
        else:
            return False

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
        tweet_text = mention['text']
        # Check if this mention has already been processed
        # TODO 2024-02-05: when processing a chat, add it to a list. if the processing is completed and not in the cached or replied to tweets, then it is not responded to.
        #  Conversely, in twitter_bot_app, reset the initial list every 5 loops
        if (not test) and (tweet_id in self.cached_already_replied_to_tweet_ids or tweet_id in self.cached_invalid_tweets or not self.check_if_valid_mention(mention_text=tweet_text)):
            logging.info(f"Mention already processed or invalid: {tweet_id}, with content [{tweet_text}]")
            return
        else:
            logging.info(f"Mention not processed yet, processing it with user input: [{tweet_text}]")

        if tweet_id in self.last_reply_times:
            first_reply_id = self.last_reply_times[user_id]
            already_processed_reply = f"Already processed this mention. See first reply: {first_reply_id}"
            self.api.update_status(status=already_processed_reply, in_reply_to_status_id=tweet_id)
            logging.info(f"Mention already processed: {tweet_id}")
            return

        if self.should_reply_to_user(user_id):
            # Check if the mention is a reply or a quote
            command, _ = extract_command_and_message(tweet_text)

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
                shared_chat_link = create_shared_chat(chat_response, metadata)
                media_id = take_screenshot_and_upload(url=f"https://www.{shared_chat_link}") if self.take_screenshot else None
                self.reply_to_tweet(user_id=user_id, response=shared_chat_link, tweet_id=tweet_id, test=test, command=command, media_id=media_id, post_reply_in_prod=post_reply_in_prod, is_paid_account=is_paid_account)
                self.last_reply_times[user_id] = time.time()  # Store the current timestamp
                self.cached_already_replied_to_tweet_ids.append(tweet_id)
            else:
                logging.error("No response generated for the mention.")
        else:
            logging.info(f"Rate limit: Not replying to {user_id}")


