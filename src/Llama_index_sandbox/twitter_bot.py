import logging
import os
import time
import tweepy
import requests
from src.Llama_index_sandbox.main import initialise_chatbot
from src.Llama_index_sandbox.retrieve import ask_questions
from src.Llama_index_sandbox.utils.utils import process_messages, get_last_index_embedding_params
from src.Llama_index_sandbox.custom_react_agent.tools.reranker.custom_query_engine import CustomQueryEngine
from src.Llama_index_sandbox.utils.gcs_utils import set_secrets_from_cloud

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set secrets from cloud
set_secrets_from_cloud()

# Twitter API credentials
consumer_key = os.environ.get('TWITTER_CONSUMER_KEY')
consumer_secret = os.environ.get('TWITTER_CONSUMER_SECRET')
access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Initialize the chatbot
engine = 'chat'
query_engine_as_tool = True
recreate_index = False
retrieval_engine, query_engine, store_response_partial, config = initialise_chatbot(
    engine=engine, query_engine_as_tool=query_engine_as_tool, recreate_index=recreate_index
)
CustomQueryEngine.load_or_compute_weights(document_weight_mappings=CustomQueryEngine.document_weight_mappings,
                                          weights_file=CustomQueryEngine.weights_file,
                                          authors_list=CustomQueryEngine.authors_list,
                                          authors_weights=CustomQueryEngine.authors_weights,
                                          recompute_weights=True)


# Function to process and respond to chat messages
def process_chat_message(message):
    try:
        response, formatted_metadata = ask_questions(
            input_queries=[message],
            retrieval_engine=retrieval_engine,
            query_engine=query_engine,
            store_response_partial=store_response_partial,
            engine=engine,
            query_engine_as_tool=query_engine_as_tool,
            chat_history=[],
            run_application=True,
            reset_chat=config.reset_chat
        )
        return response
    except Exception as e:
        logging.error(f"Error processing chat message: {e}")
        return None


# Function to get the last seen tweet ID from a file
def retrieve_last_seen_id(file_name):
    try:
        with open(file_name, 'r') as f_read:
            last_seen_id = int(f_read.read().strip())
        return last_seen_id
    except FileNotFoundError:
        logging.error("File not found. Starting from the beginning.")
        return 1


# Function to store the last seen tweet ID in a file
def store_last_seen_id(last_seen_id, file_name):
    with open(file_name, 'w') as f_write:
        f_write.write(str(last_seen_id))


# Function to fetch the entire thread above a reply
def fetch_thread(tweet_id):
    thread = []
    while tweet_id:
        try:
            tweet = api.get_status(tweet_id, tweet_mode='extended')
            thread.append(tweet.full_text)
            tweet_id = tweet.in_reply_to_status_id
        except tweepy.TweepError as e:
            logging.error(f"Error fetching tweet: {e}")
            break
    return " ".join(reversed(thread))  # Return the thread as a single string


# Function to determine if the reply is asking for 'thread' or 'tweet'
def extract_command_and_message(message):
    command = "tweet"  # default command
    if " thread" in message.lower():
        command = "thread"
    return command, message

# Function to reply to tweets
def reply_to_tweets():
    last_seen_id = retrieve_last_seen_id('last_seen_id.txt')
    mentions = api.mentions_timeline(last_seen_id, tweet_mode='extended')

    for mention in reversed(mentions):
        logging.info(f"Replying to {mention.user.screen_name} - {mention.id}")
        last_seen_id = mention.id
        store_last_seen_id(last_seen_id, 'last_seen_id.txt')

        command, user_message = extract_command_and_message(mention.full_text)

        if command == "thread":
            message = fetch_thread(mention.id)
        elif 'quoted_status' in mention._json:
            message = mention.quoted_status.full_text
        else:
            message = user_message

        # Process the message using chatbot logic
        response = process_chat_message(message)
        if response:
            reply = response
            api.update_status('@' + mention.user.screen_name + ' ' + reply, mention.id)
        else:
            logging.error("Error processing message or no response generated.")


# Main function
def main():
    while True:
        reply_to_tweets()
        time.sleep(15)  # Sleep to avoid hitting rate limits


if __name__ == "__main__":
    main()
