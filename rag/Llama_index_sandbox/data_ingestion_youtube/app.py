from flask import Flask, request, jsonify

app = Flask(__name__)

# Assuming you have your chatbot initialized as `chatbot`
# If you need to initialize your chatbot, do it here

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    # Get message from the POST request
    data = request.get_json()
    message = data.get("message")

    if not message:
        return jsonify({"error": "Message not provided"}), 400

    # Get chatbot's response
    response = chatbot.get_response(message)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
