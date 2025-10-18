from flask import Flask, request, jsonify
from ollama import chat, ChatResponse
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)


@app.route("/inference", methods=["POST"])
def inference():
    """
    Processes a user prompt and generates a response using the Mistral model.
    This endpoint accepts a JSON payload containing a 'prompt' field, passes it to 
    the Mistral model for inference, and returns the generated response in JSON format.

    Args: 
        None. Expects a JSON body with a 'prompt' field in the POST request.

    Returns:
        Response: A Flask JSON response containing:
            - 200: JSON with the model-generated response under 'response' key.
            - 400: JSON error message if 'prompt' is missing or empty.
            - 500: JSON error message if an unexpected error occurs.
    """

    try:
        data = request.get_json()
        logging.info(f"Received data: {data}")

        if "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' field"}), 400 
        
        if not data["prompt"] or not data["prompt"].strip():
            return jsonify({"error": "Prompt cannot be empty"}), 400
        

        response: ChatResponse  = chat(model="mistral", messages=[{"role": "user", "content": data["prompt"]}])
        logging.info(f"Model's response: {response}")
        
        return jsonify({
            'response': response["message"]["content"]
        }), 200

    except Exception as e:
        logging.exception("Error during inference")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)

