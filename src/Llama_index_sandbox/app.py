import json
import logging
import os
import threading
import time
import uuid
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from google.cloud import firestore

from src.Llama_index_sandbox.gcs_utils import get_firestore_client
from src.Llama_index_sandbox.main import initialise_chatbot
from src.Llama_index_sandbox.retrieve import ask_questions

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://www.mev.fyi"}})

# Initialize the chatbot
engine = 'chat'
query_engine_as_tool = True
recreate_index = False
retrieval_engine, query_engine, store_response_partial = initialise_chatbot(engine=engine, query_engine_as_tool=query_engine_as_tool, recreate_index=recreate_index)

# Initialize Firestore DB
db = get_firestore_client()


def test_pinecone_from_gcs():
    import requests
    import logging

    # Enable HTTP logging
    logging.basicConfig(level=logging.DEBUG)
    requests_logger = logging.getLogger("requests.packages.urllib3")
    requests_logger.setLevel(logging.DEBUG)
    requests_logger.propagate = True
    logging.info("Testing Pinecone from GCS")
    logging.info("Testing Pinecone from GCS")
    logging.info("Testing Pinecone from GCS")

    url = 'https://quickstart-377ec93.svc.gcp-starter.pinecone.io/query'

    # Attempt to connect using the requests library
    try:
        response = requests.get(url, timeout=5)
        logging.info(response.status_code)
    except requests.exceptions.RequestException as e:
        logging.info(e)


@app.route('/healthz')
def health():
    return 'OK', 200


def background_processing(message, job_id):
    try:
        response, formatted_metadata = ask_questions(
            input_queries=[message],
            retrieval_engine=retrieval_engine,
            query_engine=query_engine,
            store_response_partial=store_response_partial,
            engine=engine,
            query_engine_as_tool=query_engine_as_tool,
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
    test_pinecone_from_gcs()
    data = request.get_json()
    message = data.get("message")

    if not message:
        return jsonify({"error": "Message not provided"}), 400

    job_id = str(uuid.uuid4())
    thread = threading.Thread(target=background_processing, args=(message, job_id))
    thread.start()

    return jsonify({"status": "processing", "job_id": job_id}), 202


@app.route('/stream/<job_id>')
def stream(job_id):
    def generate():
        # Create an event stream from Firestore
        doc_ref = db.collection('chat_responses').document(job_id)

        # Check for response in Firestore and yield when available
        while True:
            doc = doc_ref.get()
            if doc.exists:
                yield f"data: {json.dumps(doc.to_dict())}\n\n"
                break
            time.sleep(1)
    return Response(generate(), content_type='text/event-stream')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, threaded=True)
