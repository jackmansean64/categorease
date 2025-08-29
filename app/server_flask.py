import os
from dotenv import load_dotenv

load_dotenv()

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import jinja2
import markupsafe
import xlwings as xw
from flask import Flask, Response, request, send_from_directory
from flask.templating import render_template
from flask_cors import CORS
from categorize import retrieve_transactions, categorize_transaction_batch


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

logger = logging.getLogger()
logger.setLevel(log_level)
logger.addHandler(file_handler)
logger.addHandler(console_handler)




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
def categorize_transactions_prompt():
    with xw.Book(json=request.json) as book:
        _, uncategorized_transactions = retrieve_transactions(book)
        num_uncategorized = len(uncategorized_transactions)
        book.app.alert(
            prompt=f"This will categorize {num_uncategorized} uncategorized transactions.",
            title="Are you sure?",
            buttons="ok_cancel",
            # this is the JS function name that gets called when the user clicks a button
            callback="categorizeTransactions",
        )
        return book.json()


@app.route("/categorize-transactions-batch-init", methods=["POST"])
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
            temp_sheet.range("B3").value = 5

        except Exception as e:
            logging.error(f"Error creating batch info sheet: {e}")

        return book.json()


@app.route("/categorize-transactions-batch", methods=["POST"])
def categorize_transactions_batch():
    """Process a specific batch of transactions"""
    with xw.Book(json=request.json) as book:
        try:
            temp_sheet = book.sheets["_batch_info"]
            current_batch = int(temp_sheet.range("B2").value)
            batch_size = int(temp_sheet.range("B3").value)

            book = categorize_transaction_batch(
                book, current_batch, batch_size
            )

            temp_sheet.range("B2").value = current_batch + 1

        except Exception as e:
            logging.error(f"Error in batch processing: {e}")

        return book.json()


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


# Add xlwings.html as additional source for templates so the /xlwings/alert endpoint
# will find xlwings-alert.html. "mytemplates" can be a dummy if the app doesn't use
# own templates
loader = jinja2.ChoiceLoader(
    [
        jinja2.FileSystemLoader(str(this_dir / "mytemplates")),
        jinja2.PackageLoader("xlwings", "html"),
    ]
)
app.jinja_loader = loader


# Serve static files (HTML and icons)
# This could also be handled by an external web server such as nginx, etc.
@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(this_dir, path)


@app.errorhandler(Exception)
def xlwings_exception_handler(error):
    # This handles all exceptions, so you may want to make this more restrictive
    return Response(str(error), status=500)


# if __name__ == "__main__":
#     run_kwargs = {"host": "0.0.0.0", "port": 8000, "allow_unsafe_werkzeug": True}

#     use_local_certs = os.getenv("USE_LOCAL_CERTS") == "True"
#     if use_local_certs:
#         run_kwargs.update(
#             {
#                 "certfile": str(this_dir.parent / "certs" / "localhost+2.pem"),
#                 "keyfile": str(this_dir.parent / "certs" / "localhost+2-key.pem"),
#             }
#         )

#     socketio.run(app, **run_kwargs)

if __name__ == "__main__":
    app.run(
        port=8000,
        debug=True,
        ssl_context=(
            this_dir.parent / "certs" / "localhost+2.pem",
            this_dir.parent / "certs" / "localhost+2-key.pem",
        ),
    )
