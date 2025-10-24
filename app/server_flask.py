import os
from dotenv import load_dotenv

load_dotenv()

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from flask import Flask, Response, request, send_from_directory, jsonify
from flask_cors import CORS
from categorize import parse_transactions_data, categorize_transaction_batch, MAX_TRANSACTIONS_TO_CATEGORIZE

TRANSACTION_BATCH_SIZE = 5

app = Flask(__name__)
CORS(app)

this_dir = Path(__file__).resolve().parent

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

logging.root.addHandler(file_handler)
logging.root.addHandler(console_handler)

logging.getLogger('server_flask').setLevel(log_level)
logging.getLogger('categorize').setLevel(log_level)

logger = logging.getLogger(__name__)

disable_multi_threading = os.getenv("DISABLE_MULTI_THREADING", "false").lower() == "true"
if disable_multi_threading:
    logger.info("Multi-threading disabled, processing transactions synchronously")

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

        logger.info(f"Processing batch {batch_number} with {len(transactions_data)} total transactions")

        categorized_results = categorize_transaction_batch(
            transactions_data=transactions_data,
            categories_data=categories_data,
            batch_number=batch_number,
            batch_size=batch_size
        )

        return jsonify(categorized_results)

    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route("/<path:path>")
def static_proxy(path):
    from werkzeug.exceptions import NotFound
    try:
        return send_from_directory(this_dir, path)
    except NotFound:
        # Silently ignore common missing files like favicon.ico
        if path not in ['favicon.ico']:
            logging.warning(f"File not found: {path}")
        return Response("Not found", status=404)


@app.errorhandler(Exception)
def exception_handler(error):
    logger.error(f"Unhandled exception: {error}", exc_info=True)
    return Response(str(error), status=500)


if __name__ == "__main__":
    app.run(
        port=8000,
        debug=True,
        ssl_context=(
            this_dir.parent / "certs" / "localhost+2.pem",
            this_dir.parent / "certs" / "localhost+2-key.pem",
        ),
    )
