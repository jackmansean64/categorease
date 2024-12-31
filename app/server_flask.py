from pathlib import Path

import jinja2
import markupsafe
import xlwings as xw
from flask import Flask, Response, request, send_from_directory
from flask.templating import render_template
from flask_cors import CORS

from app.categorize import read_excel_data

app = Flask(__name__)
CORS(app)

this_dir = Path(__file__).resolve().parent


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
        book.app.alert(
            prompt="This will categorize X transactions at an estimated cost of Y.",
            title="Are you sure?",
            buttons="ok_cancel",
            # this is the JS function name that gets called when the user clicks a button
            callback="categorizeTransactions",
        )
        return book.json()


@app.route("/categorize-transactions", methods=["POST"])
def categorize_transactions():
    with xw.Book(json=request.json) as book:
        transactions, categories = read_excel_data(book)
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


if __name__ == "__main__":
    app.run(
        port=8000,
        debug=True,
        ssl_context=(
            this_dir.parent / "certs" / "localhost+2.pem",
            this_dir.parent / "certs" / "localhost+2-key.pem",
        ),
    )
