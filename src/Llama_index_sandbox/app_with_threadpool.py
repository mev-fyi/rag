import json
import logging
import os
import time
import uuid
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from google.cloud import firestore
from concurrent.futures import ThreadPoolExecutor

from src.Llama_index_sandbox.utils.gcs_utils import get_firestore_client, set_secrets_from_cloud
from src.Llama_index_sandbox.main import initialise_chatbot
from src.Llama_index_sandbox.retrieve import ask_questions

set_secrets_from_cloud()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://www.mev.fyi"}})

# Setup executor for handling background tasks
executor = ThreadPoolExecutor(int(os.environ.get('NUMBER_OF_APP_WORKERS')))  # Adjust the number of workers if needed

# Initialize Firestore DB
db = get_firestore_client()

# Initialize the chatbot
engine = 'chat'
query_engine_as_tool = True
recreate_index = False
retrieval_engine, query_engine, store_response_partial = initialise_chatbot(
    engine=engine, query_engine_as_tool=query_engine_as_tool, recreate_index=recreate_index
)

logging.info("DISABLING HTTPS / SSL VERIFICATION")
os.environ['PYTHONHTTPSVERIFY'] = '0'


@app.route('/healthz')
def health():
    return 'OK', 200


def background_processing(message, chat_history, job_id):
    # This function will run in the background, invoked by the ThreadPoolExecutor
    try:
        response, formatted_metadata = ask_questions(
            input_queries=[message],
            retrieval_engine=retrieval_engine,
            query_engine=query_engine,
            store_response_partial=store_response_partial,
            engine=engine,
            query_engine_as_tool=query_engine_as_tool,
            chat_history=chat_history,
            run_application=True
        )
        logging.info(f"Job {job_id} completed successfully with response: {response} \n\n{formatted_metadata}")

        # Save the response to Firestore
        db.collection('chat_responses').document(job_id).set({
            'response': f"{response} \n\n{formatted_metadata}",
            'timestamp': firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        logging.error(f"Error processing job {job_id}: {e}")
        db.collection('chat_responses').document(job_id).set({
            'response': f"Error: {e}",
            'timestamp': firestore.SERVER_TIMESTAMP
        })


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    # This endpoint schedules background_processing to be run in the background and immediately returns a job ID
    data = request.get_json()
    message = data.get("message")
    chat_history = data.get("chat_history")

    if not message:
        return jsonify({"error": "Message not provided"}), 400

    job_id = str(uuid.uuid4())
    executor.submit(background_processing, message, chat_history, job_id)

    return jsonify({"status": "processing", "job_id": job_id}), 202


@app.route('/stream/<job_id>')
def stream(job_id):
    # This implementation stays the same, using server-sent events to stream the response
    def generate():
        doc_ref = db.collection('chat_responses').document(job_id)
        while True:
            doc = doc_ref.get()
            if doc.exists:
                yield f"data: {json.dumps(doc.to_dict())}\n\n"
                break
            time.sleep(1)

    return Response(generate(), content_type='text/event-stream')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
