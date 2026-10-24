import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

app = Flask(__name__)
CORS(app)

api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key) if api_key else None


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "PyNexa AI Backend",
        "provider": "Gemini",
        "status": "online"
    })


@app.route("/chat", methods=["POST"])
def chat():
    if client is None:
        return jsonify({
            "error": "Gemini API is not configured."
        }), 500

    data = request.get_json(silent=True) or {}
    message = data.get("message")

    if not isinstance(message, str) or not message.strip():
        return jsonify({
            "error": "Please provide a message."
        }), 400

    if len(message) > 10000:
        return jsonify({
            "error": "Your message is too long."
        }), 400

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                "You are PyNexa AI, a helpful AI assistant "
                "inside PyNexa AI Studio.\n\n"
                + message.strip()
            ]
        )

        reply = response.text

        if not reply:
            return jsonify({
                "error": "The AI returned an empty response."
            }), 502

        return jsonify({"reply": reply})

    except Exception:
        app.logger.exception("Gemini API request failed")
        return jsonify({
            "error": "The AI request failed. Please try again."
        }), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)