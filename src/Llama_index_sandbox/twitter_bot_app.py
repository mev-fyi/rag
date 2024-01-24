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
    # Here you'll process the Twitter event
    # You can use your TwitterBot class to handle the event
    twitter_data = request.json
    bot.process_webhook_data(twitter_data)
    return "200 OK"


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
