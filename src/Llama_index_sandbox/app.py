import logging
import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.utils import print_text

from src.Llama_index_sandbox import globals as glb
import src.Llama_index_sandbox.constants as constants

from src.Llama_index_sandbox.gcs_utils import get_firestore_client, set_secrets_from_cloud
from src.Llama_index_sandbox.main import initialise_chatbot
from src.Llama_index_sandbox.retrieve import ask_questions
from src.Llama_index_sandbox.utils import get_last_index_embedding_params

set_secrets_from_cloud()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://www.mev.fyi"}})

# Initialize Firestore DB
db = get_firestore_client()

# Initialize the chatbot
engine = 'chat'
query_engine_as_tool = True
recreate_index = False
retrieval_engine, query_engine, store_response_partial, config = initialise_chatbot(
    engine=engine, query_engine_as_tool=query_engine_as_tool, recreate_index=recreate_index
)


@app.route('/healthz')
def health():
    return 'OK', 200


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    # This endpoint processes the chat request synchronously and returns a response
    data = request.get_json()
    message = data.get("message")

    if not message:
        return jsonify({"error": "Message not provided"}), 400

    job_id = str(uuid.uuid4())

    try:
        response, formatted_metadata = ask_questions(
            input_queries=[message],
            retrieval_engine=retrieval_engine,
            query_engine=query_engine,
            store_response_partial=store_response_partial,
            engine=engine,
            query_engine_as_tool=query_engine_as_tool,
            run_application=True,
            reset_chat=config.reset_chat
        )
        # response.response += f"\n\n{formatted_metadata}"

        # logging.info(f"Job {job_id} completed successfully with response: {response}")

        # db.collection('chat_responses').document(job_id).set({
        #     'response': f"{response.response}",
        #     'timestamp': firestore.SERVER_TIMESTAMP
        # })
        # Save the response to Firestore
        embedding_model_name, text_splitter_chunk_size, chunk_overlap, _ = get_last_index_embedding_params()

        model_specifications = {
            "embedding_model_parameters": {
                "embedding_model_name": embedding_model_name,
                "text_splitter_chunk_size": text_splitter_chunk_size,
                "chunk_overlap": chunk_overlap,
                "number of chunks to retrieve": glb.NUMBER_OF_CHUNKS_TO_RETRIEVE,  # NOTE 2023-10-30: fix the retrieval of this as global variable
                "temperature": constants.LLM_TEMPERATURE,
            }
        }

        response_dict = {
            "status": "completed",
            "response": response,
            "formatted_metadata": formatted_metadata,
            "job_id": job_id,
            "model_specifications": model_specifications,
        }
        logging.info(f'Returning response: {response_dict}')
        if os.environ.get('ENVIRONMENT') == 'LOCAL':
            print_text(f"Final reply:\n\n{response}\n", color="pink")
        return jsonify(response_dict), 200

    except Exception as e:
        logging.error(f"Error processing job {job_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


# @app.route('/stream/<job_id>')
# def stream(job_id):
#     # This implementation can stay the same if you still want to use server-sent events
#     def generate():
#         doc_ref = db.collection('chat_responses').document(job_id)
#         while True:
#             doc = doc_ref.get()
#             if doc.exists:
#                 yield f"data: {json.dumps(doc.to_dict())}\n\n"
#                 break
#             time.sleep(1)
#
#     return Response(generate(), content_type='text/event-stream')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
