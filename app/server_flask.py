import eventlet
import os
from dotenv import load_dotenv

load_dotenv()
if os.getenv("DEBUG") != "True":
    eventlet.monkey_patch()

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import jinja2
import markupsafe
from flask import Flask, Response, request, send_from_directory, jsonify
from flask.templating import render_template
from flask_cors import CORS
from flask_socketio import SocketIO
from categorize import parse_transactions_data, categorize_transaction_batch, MAX_TRANSACTIONS_TO_CATEGORIZE

TRANSACTION_BATCH_SIZE = 5

app = Flask(__name__)
CORS(app)

this_dir = Path(__file__).resolve().parent

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    logger=True,
    # engineio_logger=True
)

disable_multi_threading = os.getenv("DISABLE_MULTI_THREADING", "false").lower() == "true"
if disable_multi_threading:
    logging.info("Multi-threading disabled, processing transactions synchronously")

env_log_level = os.getenv("LOG_LEVEL", "INFO")
log_level = getattr(logging, env_log_level.upper(), logging.INFO)

file_handler = RotatingFileHandler(
    "flask_app.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
)
file_handler.setLevel(log_level)

console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)

formatter = logging.Formatter(
    "{asctime} - {levelname} - {message}", style="{", datefmt="%Y-%m-%d %H:%M"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(log_level)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


@socketio.on("connect")
def handle_connect():
    logging.info("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    logging.info("Client disconnected")


@socketio.on("test_connection")
def handle_test_connection():
    logging.info("Test connection received")
    socketio.emit("test_response", {"message": "Test successful"})


@app.route("/")
def root():
    return {"status": "ok"}


@app.route("/categorize-transactions-prompt", methods=["POST"])
def categorize_transactions_prompt():
    """Check how many uncategorized transactions exist"""
    data = request.json
    transactions_data = data.get('transactions', [])
    categories_data = data.get('categories', [])

    _, uncategorized_transactions = parse_transactions_data(transactions_data, categories_data)
    num_uncategorized = len(uncategorized_transactions)

    return jsonify({
        'num_uncategorized': num_uncategorized,
        'exceeds_limit': num_uncategorized > MAX_TRANSACTIONS_TO_CATEGORIZE,
        'limit': MAX_TRANSACTIONS_TO_CATEGORIZE
    })


@app.route("/categorize-transactions-batch-init", methods=["POST"])
def categorize_transactions_batch_init():
    """Initialize batch processing configuration"""
    data = request.json
    transactions_data = data.get('transactions', [])
    categories_data = data.get('categories', [])

    _, uncategorized_transactions = parse_transactions_data(transactions_data, categories_data)
    total_uncategorized = len(uncategorized_transactions)

    return jsonify({
        'total_uncategorized': total_uncategorized,
        'batch_size': TRANSACTION_BATCH_SIZE,
        'transaction_limit': MAX_TRANSACTIONS_TO_CATEGORIZE
    })


@app.route("/categorize-transactions-batch", methods=["POST"])
def categorize_transactions_batch_endpoint():
    """Process a specific batch of transactions"""
    try:
        data = request.json
        transactions_data = data.get('transactions', [])
        categories_data = data.get('categories', [])
        batch_number = data.get('batch_number', 0)
        batch_size = data.get('batch_size', TRANSACTION_BATCH_SIZE)

        logging.info(f"Processing batch {batch_number} with {len(transactions_data)} total transactions")

        categorized_results = categorize_transaction_batch(
            transactions_data=transactions_data,
            categories_data=categories_data,
            socketio=socketio,
            batch_number=batch_number,
            batch_size=batch_size
        )

        return jsonify(categorized_results)

    except Exception as e:
        logging.error(f"Error in batch processing: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500




# Serve static files (HTML and icons)
# This could also be handled by an external web server such as nginx, etc.
@app.route("/<path:path>")
def static_proxy(path):
    logging.debug(f"Static file request for: {path}")
    return send_from_directory(this_dir, path)


@app.errorhandler(Exception)
def exception_handler(error):
    logging.error(f"Unhandled exception: {error}", exc_info=True)
    return Response(str(error), status=500)


if __name__ == "__main__":
    run_kwargs = {"host": "0.0.0.0", "port": 8000, "allow_unsafe_werkzeug": True}

    use_local_certs = os.getenv("USE_LOCAL_CERTS") == "True"
    if use_local_certs:
        run_kwargs.update(
            {
                "certfile": str(this_dir.parent / "certs" / "localhost+2.pem"),
                "keyfile": str(this_dir.parent / "certs" / "localhost+2-key.pem"),
            }
        )

    socketio.run(app, **run_kwargs)
