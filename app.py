import os
import json
import time
import random
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from google import genai

app = Flask(__name__)
CORS(app)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

MODEL_NAME = "gemini-3.8-flash"
def generate_stream_with_retry(message, attempts=3):
    for attempt in range(attempts):
        try:
            return client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=SYSTEM_PROMPT + "\n\nUser: " + message
            )
        except Exception as error:
            error_text = str(error)

            retryable = any(code in error_text for code in [
                "503", "UNAVAILABLE",
                "429", "RESOURCE_EXHAUSTED",
                "500", "INTERNAL",
                "502", "504"
            ])

            if not retryable or attempt == attempts - 1:
                raise

            delay = min(2 ** attempt + random.random(), 8)
            time.sleep(delay)
SYSTEM_PROMPT = (
    "You are PyNexa AI, a helpful, intelligent AI assistant "
    "inside PyNexa AI Studio. Give clear, accurate, and useful "
    "answers. Explain technical topics step by step when appropriate."
)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "PyNexa AI Backend",
        "provider": "Gemini",
        "status": "online",
        "streaming": True
    })


def validate_message():
    data = request.get_json(silent=True) or {}
    message = data.get("message")

    if not isinstance(message, str) or not message.strip():
        return None, ({
            "error": "Please provide a message."
        }, 400)

    message = message.strip()

    if len(message) > 10000:
        return None, ({
            "error": "Your message is too long."
        }, 400)

    return message, None


@app.route("/chat", methods=["POST"])
def chat():
    """Regular, non-streaming chat endpoint."""
    if client is None:
        return jsonify({
            "error": "Gemini API is not configured."
        }), 500

    message, error = validate_message()

    if error:
        return jsonify(error[0]), error[1]

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=SYSTEM_PROMPT + "\n\nUser: " + message
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


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """Stream Gemini's response as Server-Sent Events."""
    if client is None:
        return jsonify({
            "error": "Gemini API is not configured."
        }), 500

    message, error = validate_message()

    if error:
        return jsonify(error[0]), error[1]

    def generate():
        try:
            stream = generate_stream_with_retry(message)
            for chunk in stream:
                text = chunk.text

                if text:
                    yield "data: " + json.dumps({
                        "text": text
                    }) + "\n\n"

            yield "data: " + json.dumps({
                "done": True
            }) + "\n\n"

        except Exception:
            app.logger.exception(
                "Gemini streaming request failed"
            )

            yield "data: " + json.dumps({
                "error": "The AI request failed. Please try again."
            }) + "\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        threaded=True
    )
