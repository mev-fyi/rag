import threading
import time
from flask import Flask, request, jsonify
from src.Llama_index_sandbox.twitter_bot import TwitterBot

# Initialize the TwitterBot
bot = TwitterBot()

# Create a Flask app for testing
app = Flask(__name__)


@app.route('/webhook/twitter', methods=['POST'])
def twitter_webhook():
    twitter_data = request.json
    bot.process_webhook_data(twitter_data, test=True)
    return "200 OK"


# Function to simulate a webhook event for a specific tweet
def simulate_single_tweet_event():
    time.sleep(2)  # Delay to ensure the Flask app is up and running
    actual_tweet_id = '1750323891061633300'  # Actual tweet ID extracted from the URL
    sample_data_single_tweet = {
        'tweet_create_events': [
            {
                'user': {'id_str': '123456'},
                'id_str': actual_tweet_id,
                'text': "Explain tweet",
                'in_reply_to_status_id_str': None,  # Assuming this is not a reply for the simulation
                'quoted_status': None
            }
        ]
    }

    with app.test_client() as client:
        client.post('/webhook/twitter', json=sample_data_single_tweet)


# Function to simulate a webhook event for a whole thread
def simulate_thread_event():
    time.sleep(4)  # Delay to ensure the Flask app is up and running after the first event
    actual_tweet_id = '1750534894319497324'  #'1741190516341973354' # '1750534894319497324'  # '1741192495155884123'#'1750534894319497324'  # Actual tweet ID extracted from the URL
    sample_data_thread = {
        'tweet_create_events': [
            {
                'user': {'id_str': '123456'},
                'id_str': actual_tweet_id,
                'text': "Explain thread",
                'in_reply_to_status_id_str': None,  # Assuming this is the start of a thread for the simulation
                'quoted_status': None
            }
        ]
    }

    with app.test_client() as client:
        client.post('/webhook/twitter', json=sample_data_thread)


# Run the Flask app and the simulation in separate threads
if __name__ == '__main__':
    flask_thread = threading.Thread(target=lambda: app.run(port=5000))
    simulation_single_tweet_thread = threading.Thread(target=simulate_single_tweet_event)
    simulation_thread_thread = threading.Thread(target=simulate_thread_event)

    flask_thread.start()
    simulation_single_tweet_thread.start()
    # simulation_thread_thread.start()

    flask_thread.join()
    # simulation_single_tweet_thread.join()
    # simulation_thread_thread.join()
