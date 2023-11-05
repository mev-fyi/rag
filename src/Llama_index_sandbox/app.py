import logging

from src.Llama_index_sandbox.gcs_utils import set_secrets_from_cloud
from src.Llama_index_sandbox.main import initialise_chatbot
from src.Llama_index_sandbox.retrieve import ask_questions

from flask import Flask, request, jsonify

app = Flask(__name__)

engine = 'chat'
query_engine_as_tool = True
retrieval_engine, query_engine, store_response_partial = initialise_chatbot(engine=engine, query_engine_as_tool=query_engine_as_tool)
set_secrets_from_cloud()


@app.route('/hello')
def hello_world():
    return 'Hello World!'


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    # Get message from the POST request
    data = request.get_json()
    message = data.get("message")

    if not message:
        return jsonify({"error": "Message not provided"}), 400

    logging.info(f"Received message: {message}")
    # Call the ask_questions function with the message to get chatbot's response
    # Since input_queries expects a list, wrap the message in a list
    response, formatted_metadata = ask_questions(input_queries=[message], retrieval_engine=retrieval_engine,
                                                 query_engine=query_engine, store_response_partial=store_response_partial,
                                                 engine=engine, query_engine_as_tool=query_engine_as_tool, run_application=True)

    response = f"""{response} \n\n{formatted_metadata}"""
    logging.info(f"Sending response with sources: \n```\n{response}\n```\n")
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
