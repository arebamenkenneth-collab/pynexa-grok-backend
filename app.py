import os
import json
import time

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


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Primary model followed by a backup model.
MODEL_NAME = "gemini-3.6-flash"

FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
]

MODEL_CANDIDATES = [MODEL_NAME] + FALLBACK_MODELS


SYSTEM_PROMPT = (
    "You are PyNexa AI, a helpful, intelligent AI assistant "
    "inside PyNexa AI Studio. Give clear, accurate, and useful "
    "answers. Explain technical topics step by step when appropriate. "
    "Be friendly, professional, and concise when possible."
)


GENERATION_CONFIG = {
    "thinking_config": {
        "thinking_level": "LOW"
    }
}


# Only these temporary errors should trigger model fallback.
RETRYABLE_ERRORS = (
    "503",
    "UNAVAILABLE",
    "429",
    "RESOURCE_EXHAUSTED",
    "500",
    "INTERNAL",
    "502",
    "504",
    "SERVICE_UNAVAILABLE",
    "DEADLINE_EXCEEDED",
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
# ERROR HELPERS
# ============================================================

def is_retryable_error(error):
    error_text = str(error).upper()

    return any(
        code in error_text
        for code in RETRYABLE_ERRORS
    )


def log_model_failure(model, error):
    app.logger.warning(
        "Gemini model %s failed: %s",
        model,
        str(error),
    )


# ============================================================
# REGULAR GEMINI GENERATION WITH FALLBACK
# ============================================================

def generate_response_with_fallback(message):
    """
    Try the primary model first.

    If a temporary service error occurs, try the backup model.
    The Google GenAI SDK handles its own transient retries.
    """

    last_error = None

    for index, model in enumerate(MODEL_CANDIDATES):
        started_at = time.monotonic()

        try:
            app.logger.info(
                "Starting regular Gemini request using %s.",
                model,
            )

            response = client.models.generate_content(
                model=model,
                contents=(
                    SYSTEM_PROMPT
                    + "\n\nUser: "
                    + message
                ),
                config=GENERATION_CONFIG,
            )

            reply = response.text

            if not reply:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            app.logger.info(
                "Model %s completed in %.2f seconds.",
                model,
                time.monotonic() - started_at,
            )

            return reply

        except Exception as error:
            last_error = error

            app.logger.exception(
                "Regular Gemini request failed using %s.",
                model,
            )

            # Do not switch models for permanent errors.
            if not is_retryable_error(error):
                raise

            # No more models available.
            if index == len(MODEL_CANDIDATES) - 1:
                break

            app.logger.warning(
                "Temporary failure on %s. "
                "Trying fallback model %s.",
                model,
                MODEL_CANDIDATES[index + 1],
            )

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "No Gemini models are configured."
    )


# ============================================================
# STREAMING GENERATION WITH FALLBACK
# ============================================================

def generate_stream_with_fallback(message):
    """
    Try models in order.

    Fallback is allowed only before any text has been sent.
    This prevents duplicate or mixed responses in the stream.
    """

    last_error = None

    for index, model in enumerate(MODEL_CANDIDATES):
        started_at = time.monotonic()
        received_text = False

        try:
            app.logger.info(
                "Starting Gemini stream with %s.",
                model,
            )

            stream = client.models.generate_content_stream(
                model=model,
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
                        app.logger.info(
                            "First text from %s received "
                            "after %.2f seconds.",
                            model,
                            time.monotonic() - started_at,
                        )

                    received_text = True
                    yield text

            app.logger.info(
                "Streaming with %s completed in %.2f seconds.",
                model,
                time.monotonic() - started_at,
            )

            return

        except Exception as error:
            last_error = error

            app.logger.exception(
                "Gemini stream failed using %s.",
                model,
            )

            # Once text has been delivered, do not restart
            # the answer with a different model.
            if received_text:
                raise

            if not is_retryable_error(error):
                raise

            if index == len(MODEL_CANDIDATES) - 1:
                break

            app.logger.warning(
                "Temporary streaming failure on %s. "
                "Switching to fallback model %s.",
                model,
                MODEL_CANDIDATES[index + 1],
            )

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "No Gemini models are configured."
    )


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
        "fallback_models": FALLBACK_MODELS,
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

    try:
        reply = generate_response_with_fallback(message)

        return jsonify({
            "reply": reply
        })

    except Exception:
        app.logger.exception(
            "Gemini chat request failed on all available models."
        )

        return jsonify({
            "error": (
                "PyNexa AI is temporarily unavailable. "
                "Please try again shortly."
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
            for text in generate_stream_with_fallback(message):
                yield "data: " + json.dumps({
                    "text": text
                }) + "\n\n"

            yield "data: " + json.dumps({
                "done": True
            }) + "\n\n"

        except Exception:
            app.logger.exception(
                "Gemini streaming request failed on all available models."
            )

            yield "data: " + json.dumps({
                "error": (
                    "PyNexa AI is temporarily unavailable. "
                    "Please try again shortly."
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