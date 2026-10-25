import os
import json
import time
import random

from flask import (
    Flask,
    request,
    jsonify,
    Response,
    stream_with_context,
)
from flask_cors import CORS
from google import genai


# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

MODEL_NAME = "gemini-3.6-flash"

SYSTEM_PROMPT = (
    "You are PyNexa AI, a helpful, intelligent AI assistant "
    "inside PyNexa AI Studio. Give clear, accurate, and useful "
    "answers. Explain technical topics step by step when appropriate. "
    "Be friendly, professional, and concise when possible."
)

# Lower thinking effort for quicker responses.
GENERATION_CONFIG = {
    "thinking_config": {
        "thinking_level": "LOW"
    }
}

RETRYABLE_ERRORS = (
    "503",
    "UNAVAILABLE",
    "429",
    "RESOURCE_EXHAUSTED",
    "500",
    "INTERNAL",
    "502",
    "504",
)


# ============================================================
# MESSAGE VALIDATION
# ============================================================

def validate_message():
    data = request.get_json(silent=True) or {}
    message = data.get("message")

    if not isinstance(message, str) or not message.strip():
        return None, (
            {"error": "Please provide a message."},
            400,
        )

    message = message.strip()

    if len(message) > 10000:
        return None, (
            {"error": "Your message is too long."},
            400,
        )

    return message, None


# ============================================================
# GEMINI STREAM WITH RETRIES AND TIMING
# ============================================================

def generate_stream_with_retry(message, attempts=2):
    """
    Retry temporary errors only before sending any text.

    Once text has been yielded, a failed stream is not restarted,
    preventing duplicate answers.
    """

    for attempt in range(attempts):
        started_at = time.monotonic()
        received_text = False

        try:
            app.logger.info(
                "Starting Gemini stream. Attempt %s/%s.",
                attempt + 1,
                attempts,
            )

            stream = client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=(
                    SYSTEM_PROMPT
                    + "\n\nUser: "
                    + message
                ),
                config=GENERATION_CONFIG,
            )

            for chunk in stream:
                text = chunk.text

                if text:
                    if not received_text:
                        first_text_time = (
                            time.monotonic() - started_at
                        )

                        app.logger.info(
                            "Gemini first text received after %.2f seconds.",
                            first_text_time,
                        )

                    received_text = True
                    yield text

            total_time = time.monotonic() - started_at

            app.logger.info(
                "Gemini stream completed in %.2f seconds.",
                total_time,
            )

            return

        except Exception as error:
            elapsed = time.monotonic() - started_at
            error_text = str(error).upper()

            app.logger.exception(
                "Gemini stream failed after %.2f seconds.",
                elapsed,
            )

            retryable = any(
                code in error_text
                for code in RETRYABLE_ERRORS
            )

            # Never restart after text has been sent.
            if received_text:
                raise

            # Don't retry permanent errors.
            if not retryable:
                raise

            # Stop when attempts are exhausted.
            if attempt == attempts - 1:
                raise

            delay = 1 + random.random()

            app.logger.warning(
                "Temporary Gemini error. Retrying in %.2f seconds.",
                delay,
            )

            time.sleep(delay)


# ============================================================
# HOME / HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "PyNexa AI Backend",
        "provider": "Gemini",
        "status": "online",
        "streaming": True,
        "model": MODEL_NAME,
    })


# ============================================================
# REGULAR CHAT ENDPOINT
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():
    if client is None:
        return jsonify({
            "error": "Gemini API is not configured."
        }), 500

    message, error = validate_message()

    if error:
        return jsonify(error[0]), error[1]

    started_at = time.monotonic()

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=(
                SYSTEM_PROMPT
                + "\n\nUser: "
                + message
            ),
            config=GENERATION_CONFIG,
        )

        reply = response.text

        app.logger.info(
            "Regular Gemini request completed in %.2f seconds.",
            time.monotonic() - started_at,
        )

        if not reply:
            return jsonify({
                "error": "The AI returned an empty response."
            }), 502

        return jsonify({
            "reply": reply
        })

    except Exception:
        app.logger.exception(
            "Gemini chat request failed."
        )

        return jsonify({
            "error": (
                "PyNexa AI could not complete your request. "
                "Please try again."
            )
        }), 502


# ============================================================
# STREAMING CHAT ENDPOINT
# ============================================================

@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    if client is None:
        return jsonify({
            "error": "Gemini API is not configured."
        }), 500

    message, error = validate_message()

    if error:
        return jsonify(error[0]), error[1]

    def generate():
        try:
            for text in generate_stream_with_retry(message):
                yield "data: " + json.dumps({
                    "text": text
                }) + "\n\n"

            yield "data: " + json.dumps({
                "done": True
            }) + "\n\n"

        except Exception:
            app.logger.exception(
                "Gemini streaming request failed."
            )

            yield "data: " + json.dumps({
                "error": (
                    "PyNexa AI could not complete your response. "
                    "Please try again."
                )
            }) + "\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        threaded=True,
)
