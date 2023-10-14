import logging

from rag.Llama_index_sandbox.main import initialise_chatbot
from rag.Llama_index_sandbox.retrieve import ask_questions

from flask import Flask, request, jsonify

app = Flask(__name__)

engine = 'chat'
query_engine_as_tool = False
retrieval_engine, query_engine, store_response_partial = initialise_chatbot(engine=engine, query_engine_as_tool=query_engine_as_tool)


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
                                                 query_engine=query_engine, store_response_partial=store_response_partial, engine=engine, query_engine_as_tool=query_engine_as_tool)

    response = f"""{response} + \n{formatted_metadata}"""
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
