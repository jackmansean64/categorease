from typing import List, Tuple, Optional, Dict
from flask_socketio import SocketIO
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import TypeAdapter
from toolkit.language_models.token_costs import calculate_total_prompt_cost, ModelName
from xlwings import Book
import pandas as pd
from toolkit.language_models.model_connection import ChatModelsSetup
from models import Transaction, Category, CategorizedTransaction
from prompt_templates import analysis_template, serialize_categories_template
from toolkit.language_models.parallel_processing import parallel_invoke_function
import logging
import uuid
import time
from dataclasses import dataclass


@dataclass
class CategorizationSession:
    session_id: str
    uncategorized_transactions: List[Transaction]
    categorized_transactions: List[Transaction]
    categories: List[Category]
    processed_transaction_ids: set
    batch_size: int = 10
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


# Global session storage (in production, use Redis or database)
active_sessions: Dict[str, CategorizationSession] = {}


def cleanup_old_sessions():
    """Remove sessions older than 1 hour"""
    current_time = time.time()
    expired_sessions = [
        session_id for session_id, session in active_sessions.items()
        if current_time - session.created_at > 3600  # 1 hour
    ]
    for session_id in expired_sessions:
        del active_sessions[session_id]


def start_categorization_session(book: Book, socketio: SocketIO) -> str:
    """Initialize a new categorization session and return session ID"""
    cleanup_old_sessions()
    
    previously_categorized_transactions, uncategorized_transactions = retrieve_transactions(book)
    categories = retrieve_categories(book)
    
    session_id = str(uuid.uuid4())
    session = CategorizationSession(
        session_id=session_id,
        uncategorized_transactions=uncategorized_transactions,
        categorized_transactions=previously_categorized_transactions,
        categories=categories,
        processed_transaction_ids=set()
    )
    
    active_sessions[session_id] = session
    
    socketio.emit("initializeProgressBar", {"value": len(uncategorized_transactions)})
    socketio.emit("sessionStarted", {"session_id": session_id, "total_transactions": len(uncategorized_transactions)})
    
    return session_id


def get_next_batch(session_id: str, book: Book, socketio: SocketIO) -> Optional[Book]:
    """Process next batch of transactions for the given session"""
    if session_id not in active_sessions:
        socketio.emit("error", {"error": "Session not found or expired"})
        return None
    
    session = active_sessions[session_id]
    
    # Get transactions that haven't been processed yet
    remaining_transactions = [
        t for t in session.uncategorized_transactions 
        if t.transaction_id not in session.processed_transaction_ids
    ]
    
    if not remaining_transactions:
        socketio.emit("clearProgressBar")
        del active_sessions[session_id]
        return None
    
    # Process next batch
    batch = remaining_transactions[:session.batch_size]
    
    try:
        categorized_transactions_and_costs: List[Tuple[CategorizedTransaction, float]] = (
            parallel_invoke_function(
                function=model_categorize_transaction,
                variable_args=batch,
                categories=session.categories,
                categorized_transactions=session.categorized_transactions,
                socketio=socketio,
            )
        )
    except Exception as e:
        logging.error(e)
        socketio.emit("error", {"error": str(e)})
        del active_sessions[session_id]
        raise e
    
    categorized_transactions = [
        transaction for transaction, _ in categorized_transactions_and_costs
    ]
    
    # Update session state
    for transaction in batch:
        session.processed_transaction_ids.add(transaction.transaction_id)
    
    # Update the book with this batch
    updated_book = update_categories_in_sheet(book, categorized_transactions)
    
    # Emit batch completion event
    batch_cost = sum(cost for _, cost in categorized_transactions_and_costs)
    socketio.emit("batchCompleted", {
        "batch_size": len(batch),
        "batch_cost": batch_cost,
        "remaining": len(remaining_transactions) - len(batch)
    })
    
    return updated_book

def categorize_transactions_in_book(book: Book, socketio: SocketIO) -> Book:
    previously_categorized_transactions, uncategorized_transactions = (
        retrieve_transactions(book)
    )
    categories = retrieve_categories(book)
    socketio.emit("initializeProgressBar", {"value": len(uncategorized_transactions)})

    try:
        categorized_transactions_and_costs: List[Tuple[CategorizedTransaction, float]] = (
            parallel_invoke_function(
                function=model_categorize_transaction,
                variable_args=uncategorized_transactions,
                categories=categories,
                categorized_transactions=previously_categorized_transactions,
                socketio=socketio,
            )
        )
    except Exception as e:
        logging.error(e)
        socketio.emit("error", {"error": str(e)})
        raise e

    socketio.emit("clearProgressBar")
    total_cost = sum(cost for _, cost in categorized_transactions_and_costs)
    print(f"Total cost: ${total_cost:.4f}")

    categorized_transactions = [
        transaction for transaction, _ in categorized_transactions_and_costs
    ]

    return update_categories_in_sheet(book, categorized_transactions)


def model_categorize_transaction(
    transaction: Transaction,
    categories: List[Category],
    categorized_transactions: List[Transaction],
    socketio: SocketIO,
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
    socketio.emit("updateProgressBar")

    total_cost = analysis_cost + parsing_cost

    return parsed_category, total_cost


def model_analyze_transaction(
    uncategorized_transaction: Transaction,
    categories: List[Category],
    categorized_transactions: List[Transaction],
    chat_model: BaseChatModel,
    model_name: ModelName,
) -> Tuple[str, float]:
    TRANSACTION_HISTORY_LENGTH = 150
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
) -> Tuple[CategorizedTransaction, float]:
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
    categorized_transaction = CategorizedTransaction.model_validate_json(
        category_assignment_response.content
    )
    # print(assigned_category)

    total_cost = calculate_total_prompt_cost(
        category_assignment_response.response_metadata["usage"]["prompt_tokens"],
        category_assignment_response.response_metadata["usage"]["completion_tokens"],
        model_name,
    )
    # print(f"Total Cost: ${total_cost}")

    return categorized_transaction, total_cost


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
            excel_row_num = index + 2
            raise ValueError(
                f"Excel row {excel_row_num} failed validation: {str(e)}\nData: {row.to_dict()}"
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
