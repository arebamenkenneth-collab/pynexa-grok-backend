import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

api_key = os.environ.get("XAI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1"
) if api_key else None


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "PyNexa AI Backend",
        "status": "online"
    })


@app.route("/chat", methods=["POST"])
def chat():
    if client is None:
        return jsonify({
            "error": "AI service is not configured."
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
        response = client.chat.completions.create(
            model="grok-3-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are PyNexa AI, a helpful AI assistant "
                        "inside PyNexa AI Studio."
                    )
                },
                {
                    "role": "user",
                    "content": message.strip()
                }
            ]
        )

        reply = response.choices[0].message.content

        if not reply:
            return jsonify({
                "error": "The AI returned an empty response."
            }), 502

        return jsonify({"reply": reply})

    except Exception:
        app.logger.exception("Grok API request failed")

        return jsonify({
            "error": "The AI request failed. Please try again."
        }), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
