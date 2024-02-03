from flask import Flask, request, jsonify
import hashlib
import hmac
import base64
from src.Llama_index_sandbox.twitter_bot import TwitterBot
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
bot = TwitterBot()


# def process_webhook_data(self, data, test=False, test_http_request=False):  # NOTE 2024-02-02: this is
#     """
#     Processes incoming data from the webhook.
#     :param data: The data received from the webhook
#
#     Args:
#         test:
#         test_http_request:
#     """
#     # Extract relevant information from the data
#     if 'tweet_create_events' in data:
#         for event in data['tweet_create_events']:
#             user_id = event['user']['id_str']
#             if self.should_reply_to_user(user_id):
#                 tweet_id = event['id_str']
#                 tweet_text = event['text']
#
#                 # Check if the tweet is a reply or quote
#                 if 'in_reply_to_status_id_str' in event or 'quoted_status' in event:
#                     command, _ = self.extract_command_and_message(tweet_text)
#
#                     if command == "thread":
#                         message = self.fetch_thread(tweet_id, test=test, test_http_request=test_http_request)
#                     elif command == "tweet":
#                         message = self.fetch_tweet(tweet_id, test=test, test_http_request=test_http_request)
#                     else:
#                         message = tweet_text  # Default behavior
#
#                     if message is None:
#                         logging.error("Could not fetch tweet")
#                         return
#
#                     chat_input = TWITTER_THREAD_INPUT.format(user_input=tweet_text, twitter_thread=message)
#                     # TODO 2024-01-25: if the thread or tweet is referring to document existing in the database, fetch their content too.
#                     # TODO 2024-01-25: if there is one or more images to each tweet, add them.
#
#                     # Process the message
#                     response = self.process_chat_message(chat_input).response
#                     if response:
#                         self.reply_to_tweet(user_id, response, tweet_id, test)
#                         self.last_reply_times[user_id] = datetime.now()
#                     else:
#                         logging.error("No response generated for the tweet.")
#             else:
#                 logging.info(f"Rate limit: Not replying to {user_id}")
#     else:
#         logging.error("Webhook data does not contain tweet creation events.")

def verify_twitter_signature(request):
    """
    Verifies that the incoming request is from Twitter by validating its signature.
    """
    twitter_signature = request.headers.get('X-Twitter-Webhooks-Signature')
    if not twitter_signature:
        return False

    signature = 'sha256=' + base64.b64encode(hmac.new(
        key=bytes(os.environ['TWITTER_CONSUMER_SECRET'], 'utf-8'),
        msg=request.get_data(),
        digestmod=hashlib.sha256
    ).digest()).decode()

    return hmac.compare_digest(twitter_signature, signature)


@app.route('/webhook/twitter', methods=['GET'])
def twitter_crc():
    crc_token = request.args['crc_token']
    validation = hmac.new(
        key=bytes(os.environ['TWITTER_CONSUMER_SECRET'], 'utf-8'),
        msg=bytes(crc_token, 'utf-8'),
        digestmod=hashlib.sha256
    )
    signature = base64.b64encode(validation.digest())
    response = {
        'response_token': 'sha256=' + signature.decode('utf-8')
    }
    return jsonify(response)


@app.route('/webhook/twitter', methods=['POST'])
def twitter_webhook():
    # Validate the request
    if not verify_twitter_signature(request):
        return "Invalid signature", 401

    # Process the Twitter event
    twitter_data = request.json
    bot.process_webhook_data(twitter_data)
    return "200 OK"


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
