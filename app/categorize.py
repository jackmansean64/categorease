from typing import List, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import TypeAdapter
from toolkit.language_models.token_costs import calculate_total_prompt_cost, ModelName
from xlwings import Book, App
import pandas as pd
from toolkit.language_models.model_connection import ChatModelsSetup
from app.models import Transaction, Category, CategorizedTransaction
from app.prompt_templates import analysis_template, serialize_categories_template
from toolkit.language_models.parallel_processing import parallel_invoke_function


def categorize_transactions_in_book(book: Book) -> Book:
    previously_categorized_transactions, uncategorized_transactions = (
        retrieve_transactions(book)
    )
    categories = retrieve_categories(book)

    # book.app.alert("Initializing...", callback=f"initializeProgress({len(uncategorized_transactions)})")
    initialize_progress_bar_func = book.app.macro("initializeProgress")
    initialize_progress_bar_func(len(uncategorized_transactions))
    print(book.app.status_bar)
    book.app.status_bar = 5
    with book.app.properties(status_bar='Calculating...'):
        print("testing")

    # categorized_transactions_and_costs: List[Tuple[CategorizedTransaction, float]] = (
    #     parallel_invoke_function(
    #         function=model_categorize_transaction,
    #         variable_args=uncategorized_transactions,
    #         categories=categories,
    #         categorized_transactions=previously_categorized_transactions,
    #         book=book,
    #     )
    # )
    # book.app.alert("Complete!", callback="resetProgress")
    #
    # total_cost = sum(cost for _, cost in categorized_transactions_and_costs)
    # print(f"Total cost: ${total_cost:.4f}")
    #
    # categorized_transactions = [
    #     transaction for transaction, _ in categorized_transactions_and_costs
    # ]
    #
    # return update_categories_in_sheet(book, categorized_transactions)


def model_categorize_transaction(
    transaction: Transaction,
    categories: List[Category],
    categorized_transactions: List[Category],
    book: Book,
) -> Tuple[CategorizedTransaction, float]:
    chat_models = ChatModelsSetup()

    analysis_response, analysis_cost = model_analyze_transaction(
        transaction,
        categories,
        categorized_transactions,
        chat_models.claude_35_haiku_chat,
        ModelName.HAIKU_3_5,
    )

    parsed_category, parsing_cost = model_parse_category_from_analysis(
        transaction,
        analysis_response,
        chat_models.claude_35_haiku_chat,
        ModelName.HAIKU_3_5,
    )

    book.app.alert("Processing...", callback="incrementProgress")
    total_cost = analysis_cost + parsing_cost

    return parsed_category, total_cost


def model_analyze_transaction(
    uncategorized_transaction: Transaction,
    categories: List[Category],
    categorized_transactions: List[Transaction],
    chat_model: BaseChatModel,
    model_name: ModelName,
) -> Tuple[str, float]:
    TRANSACTION_HISTORY_LENGTH = 200
    prompt_template = PromptTemplate.from_template(analysis_template)

    formatted_prompt = prompt_template.format(
        categories=TypeAdapter(List[Category]).dump_python(categories),
        examples=TypeAdapter(List[Transaction]).dump_python(
            categorized_transactions[:TRANSACTION_HISTORY_LENGTH]
        ),
        transaction=uncategorized_transaction.model_dump(),
    )
    # print(formatted_prompt)

    analysis_response = chat_model.invoke(formatted_prompt)

    total_cost = calculate_total_prompt_cost(
        analysis_response.response_metadata["usage"]["prompt_tokens"],
        analysis_response.response_metadata["usage"]["completion_tokens"],
        model_name,
    )
    # print(f"Total Cost: ${total_cost}")

    return analysis_response.content, total_cost


def model_parse_category_from_analysis(
    uncategorized_transaction: Transaction,
    analysis_response: str,
    chat_model: BaseChatModel,
    model_name: ModelName,
) -> Tuple[Category, float]:
    serialization_prompt_template = PromptTemplate.from_template(
        serialize_categories_template
    )

    formatted_prompt = serialization_prompt_template.format(
        transaction=uncategorized_transaction,
        json_structure=CategorizedTransaction.model_json_schema(),
    )

    prompt = [
        AIMessage(content=analysis_response),
        HumanMessage(content=formatted_prompt),
    ]

    category_assignment_response = chat_model.invoke(prompt)
    assigned_category = CategorizedTransaction.model_validate_json(
        category_assignment_response.content
    )
    # print(assigned_category)

    total_cost = calculate_total_prompt_cost(
        category_assignment_response.response_metadata["usage"]["prompt_tokens"],
        category_assignment_response.response_metadata["usage"]["completion_tokens"],
        model_name,
    )
    # print(f"Total Cost: ${total_cost}")

    return assigned_category, total_cost


def retrieve_transactions(book: Book) -> Tuple[List[Transaction], List[Transaction]]:
    transaction_columns = [
        "Date",
        "Description",
        "Category",
        "Amount",
        "Labels",
        "Notes",
        "Account",
        "Account #",
        "Institution",
        "Month",
        "Week",
        "Transaction ID",
        "Account ID",
        "Check Number",
        "Full Description",
        "Date Added",
    ]

    transactions_sheet = book.sheets["Transactions"]
    transactions_data = transactions_sheet.range("A1").expand().value
    transactions_df = pd.DataFrame(transactions_data[1:], columns=transactions_data[0])

    transactions_df["Date"] = pd.to_datetime(transactions_df["Date"])
    transactions_df["Date Added"] = pd.to_datetime(transactions_df["Date Added"])
    transactions_df = transactions_df.astype(
        {
            "Account #": str,
            "Transaction ID": str,
            "Account ID": str,
            "Check Number": str,
        }
    )
    for col in transaction_columns:
        if col not in transactions_df.columns:
            transactions_df[col] = None
    transactions_df = transactions_df[transaction_columns]

    transactions = _convert_df_to_transactions(transactions_df)

    categorized_transactions = [
        transaction for transaction in transactions if transaction.category
    ]
    uncategorized_transactions = [
        transaction for transaction in transactions if not transaction.category
    ]
    return categorized_transactions, uncategorized_transactions


def retrieve_categories(book: Book) -> List[Category]:
    category_columns = ["Category", "Group", "Type"]

    categories_sheet = book.sheets["Categories"]
    categories_data = categories_sheet.range("A1").expand().value
    categories_df = pd.DataFrame(categories_data[1:], columns=categories_data[0])
    for col in category_columns:
        if col not in categories_df.columns:
            categories_df[col] = None
    categories_df = categories_df[category_columns]
    return _convert_df_to_categories(categories_df)


def clean_amount(amount):
    if pd.isna(amount):
        return None
    if isinstance(amount, str):
        # Remove currency symbol and commas
        cleaned = amount.replace("$", "").replace(",", "").strip()
        return float(cleaned)
    return float(amount)


def _convert_df_to_transactions(df: pd.DataFrame) -> List[Transaction]:
    df = df.copy()

    # Clean amount column first
    df["Amount"] = df["Amount"].apply(clean_amount)

    # Clean all optional columns
    columns_to_clean = [
        "Category",
        "Labels",
        "Notes",
        "Check Number",
        "Account",
        "Account #",
        "Institution",
        "Month",
        "Week",
        "Transaction ID",
        "Account ID",
        "Full Description",
    ]
    for col in columns_to_clean:
        df[col] = df[col].where(pd.notna(df[col]), None)

    transactions = []
    for index, row in df.iterrows():
        try:
            transaction = Transaction(**row.to_dict())
            transactions.append(transaction)
        except Exception as e:
            raise ValueError(
                f"Row {index} failed validation: {str(e)}\nData: {row.to_dict()}"
            )

    return transactions


def _convert_df_to_categories(df: pd.DataFrame) -> List[Category]:
    df = df.copy()

    # Clean optional columns
    columns_to_clean = ["Group", "Type"]
    for col in columns_to_clean:
        df[col] = df[col].where(pd.notna(df[col]), None)

    categories = []
    for index, row in df.iterrows():
        try:
            category = Category(**row.to_dict())
            categories.append(category)
        except Exception as e:
            raise ValueError(
                f"Row {index} failed validation: {str(e)}\nData: {row.to_dict()}"
            )

    return categories


def update_categories_in_sheet(
    book: Book, categorized_transactions: List[CategorizedTransaction]
) -> Book:
    """
    Updates the categories in the transactions sheet for the categorized transactions.
    If the transaction already has a category it will not be overwritten.
    """
    sheet = book.sheets["Transactions"]
    headers = sheet.range("A1").expand("right").value
    transaction_id_col = headers.index("Transaction ID") + 1
    category_col = headers.index("Category") + 1

    rows = sheet.tables[0].data_body_range.rows

    for transaction in categorized_transactions:
        for i, row in enumerate(rows):
            if row[transaction_id_col - 1].value == transaction.transaction_id:
                if (
                    not row[category_col - 1].value
                    and transaction.category != "Unknown"
                ):
                    sheet.cells(i + 2, category_col).value = transaction.category
                break

    return book
