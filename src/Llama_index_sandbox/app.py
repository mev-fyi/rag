import logging
import threading
import uuid
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import firestore

from src.Llama_index_sandbox.gcs_utils import set_secrets_from_cloud, get_firestore_client
from src.Llama_index_sandbox.main import initialise_chatbot
from src.Llama_index_sandbox.retrieve import ask_questions


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://www.mev.fyi"}})

engine = 'chat'
query_engine_as_tool = True
recreate_index = False
retrieval_engine, query_engine, store_response_partial = initialise_chatbot(engine=engine, query_engine_as_tool=query_engine_as_tool, recreate_index=recreate_index)
set_secrets_from_cloud()
db = get_firestore_client()  # Initialize Firestore DB


@app.route('/healthz')
def health():
    # Optionally add any vital health checks your app needs.
    return 'OK', 200


def background_processing(message, job_id):
    try:
        response, formatted_metadata = ask_questions(input_queries=[message], retrieval_engine=retrieval_engine,
                                                     query_engine=query_engine, store_response_partial=store_response_partial,
                                                     engine=engine, query_engine_as_tool=query_engine_as_tool, run_application=True)

        response = f"{response} \n\n{formatted_metadata}"
        logging.info(f"Completed processing for job {job_id}. Storing response in Firestore.")

        # Save the response to Firestore
        doc_ref = db.collection('chat_responses').document(job_id)
        doc_ref.set({
            'response': response,
            'timestamp': firestore.SERVER_TIMESTAMP
        })

    except Exception as e:
        logging.error(f"Error processing job {job_id}: {e}")


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    # Get message from the POST request
    data = request.get_json()
    message = data.get("message")

    if not message:
        return jsonify({"error": "Message not provided"}), 400

    logging.info(f"Received message: {message}")

    # Generate a unique job ID
    job_id = str(uuid.uuid4())

    # Spawn a new thread for the long-running task
    thread = threading.Thread(target=background_processing, args=(message, job_id))
    thread.start()

    # Immediately acknowledge the request
    return jsonify({"status": "processing", "job_id": job_id}), 202


@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    try:
        doc_ref = db.collection('chat_responses').document(job_id)
        doc = doc_ref.get()
        if doc.exists:
            return jsonify({"response": doc.to_dict()}), 200
        else:
            return jsonify({"error": "Result not found or still processing"}), 404
    except Exception as e:
        logging.error(f"Error retrieving job {job_id}: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
