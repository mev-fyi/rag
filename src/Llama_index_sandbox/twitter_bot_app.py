from flask import Flask, request, jsonify
import hashlib
import hmac
import base64
from twitter_bot import TwitterBot  # Assuming TwitterBot is your class
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
bot = TwitterBot()


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