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
import xlwings as xw
from flask import Flask, Response, request, send_from_directory, jsonify, g
from flask.templating import render_template
from flask_cors import CORS
from flask_socketio import SocketIO
from categorize import retrieve_transactions, categorize_transaction_batch, MAX_TRANSACTIONS_TO_CATEGORIZE
from auth import init_cognito, auth_required, optional_auth, usage_required, get_current_user, get_user_subscription_status, consume_user_transactions
from stripe_integration import stripe_service, create_subscription_checkout, create_customer_portal_url, track_transaction_usage, get_plan_info
from mongo_models import User

TRANSACTION_BATCH_SIZE = 5

app = Flask(__name__)
CORS(app)

init_cognito(app)

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


@app.route("/hello", methods=["POST"])
def hello():
    # Instantiate a Book object with the deserialized request body
    with xw.Book(json=request.json) as book:
        # Use xlwings as usual
        sheet = book.sheets[0]
        cell = sheet["A1"]
        if cell.value == "Hello xlwings!":
            cell.value = "Bye xlwings!"
        else:
            cell.value = "Hello xlwings!"

        # Pass the following back as the response
        return book.json()


@app.route("/categorize-transactions-prompt", methods=["POST"])
@optional_auth
def categorize_transactions_prompt():
    with xw.Book(json=request.json) as book:
        _, uncategorized_transactions = retrieve_transactions(book)
        num_uncategorized = len(uncategorized_transactions)
        
        if num_uncategorized > MAX_TRANSACTIONS_TO_CATEGORIZE:
            book.app.alert(
                prompt=f"Found {num_uncategorized} uncategorized transactions. Only {MAX_TRANSACTIONS_TO_CATEGORIZE} transactions can be processed at a time. Click OK to process the first {MAX_TRANSACTIONS_TO_CATEGORIZE} transactions, or Cancel to abort.",
                title="Batch Size Limit",
                buttons="ok_cancel",
                callback="categorizeTransactions",
            )
        else:
            book.app.alert(
                prompt=f"This will categorize {num_uncategorized} uncategorized transactions.",
                title="Are you sure?",
                buttons="ok_cancel",
                callback="categorizeTransactions",
            )
        return book.json()


@app.route("/categorize-transactions-batch-init", methods=["POST"])
@auth_required
@usage_required()
def categorize_transactions_batch_init():
    """Initialize batch processing and store batch info in Excel"""
    with xw.Book(json=request.json) as book:
        previously_categorized, uncategorized_transactions = retrieve_transactions(book)
        total_uncategorized = len(uncategorized_transactions)

        # Store batch info in a hidden sheet that client can read
        try:
            # Try to delete existing temp sheet
            try:
                book.sheets["_batch_info"].delete()
            except:
                pass

            temp_sheet = book.sheets.add("_batch_info")
            temp_sheet.range("A1").value = "total_uncategorized"
            temp_sheet.range("B1").value = total_uncategorized
            temp_sheet.range("A2").value = "current_batch"
            temp_sheet.range("B2").value = 0
            temp_sheet.range("A3").value = "batch_size"
            temp_sheet.range("B3").value = TRANSACTION_BATCH_SIZE

        except Exception as e:
            logging.error(f"Error creating batch info sheet: {e}")

        return book.json()


@app.route("/categorize-transactions-batch", methods=["POST"])
@auth_required
@usage_required()
def categorize_transactions_batch():
    """Process a specific batch of transactions"""
    user = get_current_user()
    
    with xw.Book(json=request.json) as book:
        try:
            temp_sheet = book.sheets["_batch_info"]
            current_batch = int(temp_sheet.range("B2").value)
            batch_size = int(temp_sheet.range("B3").value)
            
            # Get actual transaction count before processing
            _, uncategorized_transactions = retrieve_transactions(book)
            actual_transaction_count = min(batch_size, len(uncategorized_transactions))
            
            # Check and consume user transactions before processing
            if not consume_user_transactions(user, actual_transaction_count):
                return jsonify({
                    'error': 'Insufficient transaction credits',
                    'transactions_remaining': user.transactions_remaining,
                    'subscription_tier': user.subscription_tier
                }), 403
            
            # Process the batch
            book = categorize_transaction_batch(
                book, socketio, current_batch, batch_size
            )
            
            # Track usage for billing
            track_transaction_usage(user, actual_transaction_count)
            
            temp_sheet.range("B2").value = current_batch + 1

        except Exception as e:
            logging.error(f"Error in batch processing: {e}")
            # If processing failed, refund the user's transactions
            if 'actual_transaction_count' in locals():
                user.transactions_remaining += actual_transaction_count
                user.transactions_used_this_period -= actual_transaction_count
                user.save()
                logging.info(f"Refunded {actual_transaction_count} transactions to user {user.email}")

        return book.json()


@app.route("/api/auth/status", methods=["GET"])
@optional_auth
def auth_status():
    """Get current authentication and subscription status"""
    user = get_current_user()
    return jsonify(get_user_subscription_status(user))


@app.route("/api/subscription/plans", methods=["GET"])
def get_plans():
    """Get available subscription plans"""
    return jsonify(get_plan_info())


@app.route("/api/subscription/checkout", methods=["POST"])
@auth_required
def create_checkout():
    """Create Stripe checkout session"""
    data = request.get_json()
    plan_type = data.get('plan_type')
    success_url = data.get('success_url', request.url_root + 'subscription-success')
    cancel_url = data.get('cancel_url', request.url_root + 'subscription-cancel')
    
    if not plan_type or plan_type not in ['basic', 'premium']:
        return jsonify({'error': 'Invalid plan type'}), 400
    
    user = get_current_user()
    checkout_url = create_subscription_checkout(user, plan_type, success_url, cancel_url)
    
    if checkout_url:
        return jsonify({'checkout_url': checkout_url})
    else:
        return jsonify({'error': 'Failed to create checkout session'}), 500


@app.route("/api/subscription/portal", methods=["POST"])
@auth_required
def customer_portal():
    """Create customer portal session"""
    data = request.get_json()
    return_url = data.get('return_url', request.url_root)
    
    user = get_current_user()
    portal_url = create_customer_portal_url(user, return_url)
    
    if portal_url:
        return jsonify({'portal_url': portal_url})
    else:
        return jsonify({'error': 'Failed to create portal session'}), 500


@app.route("/api/webhooks/stripe", methods=["POST"])
def stripe_webhook():
    """Handle Stripe webhooks"""
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    
    result = stripe_service.handle_webhook(payload, sig_header)
    
    if result['status'] == 'success':
        return jsonify(result), 200
    else:
        return jsonify(result), 400


@app.route("/xlwings/alert")
def alert():
    """Boilerplate required by book.app.alert() and to show unhandled exceptions"""
    prompt = request.args.get("prompt")
    title = request.args.get("title")
    buttons = request.args.get("buttons")
    mode = request.args.get("mode")
    callback = request.args.get("callback")
    return render_template(
        "xlwings-alert.html",
        prompt=markupsafe.escape(prompt).replace("\n", markupsafe.Markup("<br>")),
        title=title,
        buttons=buttons,
        mode=mode,
        callback=callback,
    )


loader = jinja2.ChoiceLoader(
    [
        jinja2.FileSystemLoader(str(this_dir / "mytemplates")),
        jinja2.PackageLoader("xlwings", "html"),
    ]
)
app.jinja_loader = loader


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(this_dir, path)


@app.errorhandler(Exception)
def xlwings_exception_handler(error):
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
