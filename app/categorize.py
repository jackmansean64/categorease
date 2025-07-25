from typing import List, Tuple
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

TRANSACTION_HISTORY_LENGTH = 200


def categorize_transactions_batch_in_book(
    book: Book, 
    socketio: SocketIO, 
    batch_number: int, 
    batch_size: int
) -> Book:
    """Process a specific batch of transactions"""
    
    # Get all transactions (we need this to maintain context for AI)
    previously_categorized_transactions, uncategorized_transactions = retrieve_transactions(book)
    
    # Calculate batch boundaries
    start_idx = batch_number * batch_size
    end_idx = min(start_idx + batch_size, len(uncategorized_transactions))
    batch_transactions = uncategorized_transactions[start_idx:end_idx]
    
    if not batch_transactions:
        # No transactions to process in this batch
        logging.info(f"Batch {batch_number}: No transactions to process")
        return book
    
    categories = retrieve_categories(book)
    
    logging.info(f"Processing batch {batch_number}: transactions {start_idx} to {end_idx-1} ({len(batch_transactions)} transactions)")
    
    try:
        categorized_transactions_and_costs: List[Tuple[CategorizedTransaction, float]] = (
            parallel_invoke_function(
                function=model_categorize_transaction,
                variable_args=batch_transactions,
                categories=categories,
                categorized_transactions=previously_categorized_transactions[:TRANSACTION_HISTORY_LENGTH],
                socketio=socketio,
            )
        )
    
    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        socketio.emit("error", {"error": str(e)})
        raise e

    total_cost = sum(cost for _, cost in categorized_transactions_and_costs)
    logging.info(f"Batch {batch_number} cost: ${total_cost:.4f}")

    categorized_transactions = [
        transaction for transaction, _ in categorized_transactions_and_costs
    ]

    # Update only the transactions from this batch
    return update_categories_in_sheet_batch(book, categorized_transactions, uncategorized_transactions, start_idx)


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

    total_cost = analysis_cost + parsing_cost

    return parsed_category, total_cost


def model_analyze_transaction(
    uncategorized_transaction: Transaction,
    categories: List[Category],
    categorized_transactions: List[Transaction],
    chat_model: BaseChatModel,
    model_name: ModelName,
) -> Tuple[str, float]:
    prompt_template = PromptTemplate.from_template(analysis_template)

    formatted_prompt = prompt_template.format(
        categories=TypeAdapter(List[Category]).dump_python(categories),
        examples=TypeAdapter(List[Transaction]).dump_python(categorized_transactions),
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
        if transaction.transaction_id is None:
            print(f"No transaction ID present for {transaction.description}, category can't be assigned.")
            continue
            
        for i, row in enumerate(rows):
            if str(row[transaction_id_col - 1].value) == str(transaction.transaction_id):
                if (
                    not row[category_col - 1].value
                    and transaction.category != "Unknown"
                ):
                    sheet.cells(i + 2, category_col).value = transaction.category
                break

    return book


def update_categories_in_sheet_batch(
    book: Book, 
    categorized_transactions: List[CategorizedTransaction], 
    all_uncategorized_transactions: List[Transaction],
    start_idx: int
) -> Book:
    """Update Excel sheet with categorized transactions from a specific batch"""
    
    if not categorized_transactions:
        return book
    
    sheet = book.sheets["Transactions"]
    headers = sheet.range("A1").expand("right").value
    transaction_id_col = headers.index("Transaction ID") + 1
    category_col = headers.index("Category") + 1
    
    rows = sheet.tables[0].data_body_range.rows
    
    # Create a map of transaction IDs to their new categories for quick lookup
    transaction_id_to_category = {
        str(transaction.transaction_id): transaction.category 
        for transaction in categorized_transactions
        if transaction.transaction_id is not None and transaction.category != "Unknown"
    }
    
    updated_count = 0
    for transaction in categorized_transactions:
        if transaction.transaction_id is None:
            logging.warning(f"No transaction ID present for {transaction.description}, category can't be assigned.")
            continue
            
        transaction_id_str = str(transaction.transaction_id)
        if transaction_id_str not in transaction_id_to_category:
            continue
            
        # Find the row with this transaction ID
        for i, row in enumerate(rows):
            if str(row[transaction_id_col - 1].value) == transaction_id_str:
                # Only update if the category is currently empty
                if not row[category_col - 1].value:
                    sheet.cells(i + 2, category_col).value = transaction.category
                    updated_count += 1
                    logging.debug(f"Updated row {i + 2} with category: {transaction.category}")
                break
    
    logging.info(f"Batch update complete: {updated_count} transactions updated with categories")
    return book
