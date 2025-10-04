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
from bs4 import BeautifulSoup
import os

TRANSACTION_HISTORY_LENGTH = 150
INVALID_CATEGORY = "Invalid"
UNKNOWN_CATEGORY = "Unknown"
MAX_TRANSACTIONS_TO_CATEGORIZE = 100

processed_transaction_ids = set()

def reset_categorization_session():
    """Reset the in-memory tracking of processed transactions for a new categorization session"""
    processed_transaction_ids.clear()
    logging.info(
        "Categorization session reset - all transactions can be reprocessed"
    )


def categorize_transaction_batch(
    book: Book, socketio: SocketIO, batch_number: int, batch_size: int
) -> Book:
    """Process a specific batch of transactions"""

    previously_categorized_transactions, uncategorized_transactions = (
        retrieve_transactions(book)
    )

    unprocessed_transactions = [
        t for t in uncategorized_transactions if t.transaction_id not in processed_transaction_ids
    ]

    try:
        batch_info_sheet = book.sheets["_batch_info"]
        total_processed = int(batch_info_sheet.range("B4").value or 0)
    except:
        logging.error("Could not read total processed from _batch_info sheet")
        total_processed = 0
    
    remaining_limit = MAX_TRANSACTIONS_TO_CATEGORIZE - total_processed
    
    actual_batch_size = min(batch_size, remaining_limit, len(unprocessed_transactions))
    batch_transactions = unprocessed_transactions[:actual_batch_size]

    if not batch_transactions:
        logging.info(f"Batch {batch_number}: No unprocessed transactions remaining")
        reset_categorization_session()
        return book

    categories = retrieve_categories(book)

    logging.info(
        f"Processing batch {batch_number}: processing {len(batch_transactions)} transactions ({len(unprocessed_transactions)} unprocessed from {len(uncategorized_transactions)} total uncategorized)"
    )

    try:
        disable_multi_threading = os.getenv("DISABLE_MULTI_THREADING", "false").lower() == "true"
        
        if disable_multi_threading:
            categorized_transactions_and_costs = [
                model_categorize_transaction(
                    transaction=transaction,
                    categories=categories,
                    categorized_transactions=previously_categorized_transactions[:TRANSACTION_HISTORY_LENGTH],
                    socketio=socketio,
                )
                for transaction in batch_transactions
            ]
        else:
            categorized_transactions_and_costs = parallel_invoke_function(
                function=model_categorize_transaction,
                variable_args=batch_transactions,
                categories=categories,
                categorized_transactions=previously_categorized_transactions[:TRANSACTION_HISTORY_LENGTH],
                socketio=socketio,
            )
        
    except Exception as e:
        logging.error(f"Batch {batch_number}: Processing failed with error: {e}")
        socketio.emit("error", {"error": str(e)})
        raise e

    for transaction in batch_transactions:
        if transaction.transaction_id:
            processed_transaction_ids.add(transaction.transaction_id)

    total_cost = sum(cost for _, cost in categorized_transactions_and_costs)
    logging.info(f"Batch {batch_number} cost: ${total_cost:.4f}")

    categorized_transactions = [
        transaction for transaction, _ in categorized_transactions_and_costs
    ]

    # Update total_processed count in batch info sheet
    try:
        batch_info_sheet = book.sheets["_batch_info"]
        new_total_processed = total_processed + len(batch_transactions)
        batch_info_sheet.range("B4").value = new_total_processed
        logging.info(f"Batch {batch_number}: Updated total_processed to {new_total_processed}")
    except Exception as e:
        logging.warning(f"Failed to update total_processed count: {e}")

    return update_categories_in_sheet_batch(
        book, categorized_transactions, uncategorized_transactions
    )


def model_categorize_transaction(
    transaction: Transaction,
    categories: List[Category],
    categorized_transactions: List[Transaction],
    socketio: SocketIO,
) -> Tuple[CategorizedTransaction, float]:
    import time
    
    transaction_start_time = time.time()
    transaction_id = getattr(transaction, 'transaction_id', 'unknown')
    
    chat_models = ChatModelsSetup()

    valid_categories = [category.category for category in categories] + [
        UNKNOWN_CATEGORY
    ]

    total_cost = 0.0
    max_retries = 2

    for attempt in range(max_retries + 1):
        analysis_response, analysis_cost = model_analyze_transaction(
            transaction,
            categories,
            categorized_transactions,
            chat_models.claude_35_haiku_chat,
            ModelName.HAIKU_3_5,
        )

        total_cost += analysis_cost

        parsed_category = parse_category_from_analysis(
            transaction,
            analysis_response,
            valid_categories,
        )

        if parsed_category.category != INVALID_CATEGORY:
            transaction_time = time.time() - transaction_start_time
            logging.warning(f"Successfully categorized transaction {transaction_id} in {transaction_time:.1f}s after {attempt + 1} attempt(s)")
            return parsed_category, total_cost
        elif attempt == max_retries:
            parsed_category.category = UNKNOWN_CATEGORY
            transaction_time = time.time() - transaction_start_time
            logging.warning(
                f"Final attempt for transaction {transaction_id}, returning category: {parsed_category.category} after {transaction_time:.1f}s"
            )
            return parsed_category, total_cost
        else:
            logging.info(
                f"Attempt {attempt + 1} failed for transaction {transaction_id}, retrying..."
            )

    transaction_time = time.time() - transaction_start_time
    logging.warning(f"Completed transaction {transaction_id} in {transaction_time:.1f}s")
    return parsed_category, total_cost


def model_analyze_transaction(
    uncategorized_transaction: Transaction,
    categories: List[Category],
    categorized_transactions: List[Transaction],
    chat_model: BaseChatModel,
    model_name: ModelName,
) -> Tuple[str, float]:
    import time
    
    api_start_time = time.time()
    transaction_id = getattr(uncategorized_transaction, 'transaction_id', 'unknown')
    
    prompt_template = PromptTemplate.from_template(analysis_template)

    formatted_prompt = prompt_template.format(
        categories=TypeAdapter(List[Category]).dump_python(categories),
        examples=TypeAdapter(List[Transaction]).dump_python(categorized_transactions),
        transaction=uncategorized_transaction.model_dump(),
    )
    # print("Formatted Prompt: " + formatted_prompt)

    analysis_response = chat_model.invoke(formatted_prompt)
    api_time = time.time() - api_start_time
    
    logging.warning(f"LLM API call for transaction {transaction_id} completed in {api_time:.1f}s")

    total_cost = calculate_total_prompt_cost(
        analysis_response.response_metadata["usage"]["prompt_tokens"],
        analysis_response.response_metadata["usage"]["completion_tokens"],
        model_name,
    )
    # print(f"Total Cost: ${total_cost}")

    return analysis_response.content, total_cost


def parse_category_from_analysis(
    uncategorized_transaction: Transaction,
    analysis_response: str,
    valid_categories: List[str],
) -> CategorizedTransaction:
    soup = BeautifulSoup(analysis_response, "html.parser")
    assigned_category_tag = soup.find("assigned_category")

    if assigned_category_tag:
        category = assigned_category_tag.get_text().strip()
    else:
        import re

        match = re.search(
            r"<assigned_category>(.*?)</assigned_category>",
            analysis_response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            category = match.group(1).strip()
        else:
            logging.warning(
                f"No assigned_category tag found in analysis response for transaction {uncategorized_transaction.transaction_id}"
            )
            category = UNKNOWN_CATEGORY

    if category not in valid_categories:
        logging.warning(
            f"Invalid category '{category}' for transaction {uncategorized_transaction.transaction_id}."
        )
        category = INVALID_CATEGORY

    categorized_transaction = CategorizedTransaction(
        transaction_id=(
            str(uncategorized_transaction.transaction_id)
            if uncategorized_transaction.transaction_id
            else ""
        ),
        date=uncategorized_transaction.date,
        description=uncategorized_transaction.description,
        category=category,
    )
    logging.info(f"Successfuly categorized transaction: {categorized_transaction}")

    return categorized_transaction


def retrieve_transactions(book: Book) -> Tuple[List[Transaction], List[Transaction]]:
    transaction_columns = [
        "Date",
        "Description",
        "Category",
        "Amount",
        "Account",
        "Transaction ID",
    ]

    transactions_sheet = book.sheets["Transactions"]
    transactions_data = transactions_sheet.range("A1").expand().value
    transactions_df = pd.DataFrame(transactions_data[1:], columns=transactions_data[0])

    transactions_df["Date"] = pd.to_datetime(transactions_df["Date"])
    transactions_df = transactions_df.astype(
        {
            "Transaction ID": str,
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
        cleaned = amount.replace("$", "").replace(",", "").strip()
        return float(cleaned)
    return float(amount)


def _convert_df_to_transactions(df: pd.DataFrame) -> List[Transaction]:
    df = df.copy()

    df["Amount"] = df["Amount"].apply(clean_amount)

    columns_to_clean = [
        "Category",
        "Account",
        "Transaction ID",
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
            print(
                f"No transaction ID present for {transaction.description}, category can't be assigned."
            )
            continue

        for i, row in enumerate(rows):
            if str(row[transaction_id_col - 1].value) == str(
                transaction.transaction_id
            ):
                if (
                    not row[category_col - 1].value
                    and transaction.category != UNKNOWN_CATEGORY
                ):
                    sheet.cells(i + 2, category_col).value = transaction.category
                break
            else:
                logging.warning(
                    f"categorized transaction with ID: {transaction.transaction_id} couldn't be matched to a transaction in the sheet."
                )

    return book


def update_categories_in_sheet_batch(
    book: Book,
    categorized_transactions: List[CategorizedTransaction],
    all_uncategorized_transactions: List[Transaction],
) -> Book:
    """Update Excel sheet with categorized transactions from a specific batch"""

    if not categorized_transactions:
        return book

    sheet = book.sheets["Transactions"]
    headers = sheet.range("A1").expand("right").value
    transaction_id_col = headers.index("Transaction ID") + 1
    category_col = headers.index("Category") + 1

    rows = sheet.tables[0].data_body_range.rows

    transaction_id_to_category = {
        str(transaction.transaction_id): transaction.category
        for transaction in categorized_transactions
        if transaction.transaction_id is not None
        and transaction.category != UNKNOWN_CATEGORY
    }

    updated_count = 0
    for transaction in categorized_transactions:
        if transaction.transaction_id is None:
            logging.warning(
                f"No transaction ID present for {transaction.description}, category can't be assigned."
            )
            continue

        transaction_id_str = str(transaction.transaction_id)
        if transaction_id_str not in transaction_id_to_category:
            continue

        for i, row in enumerate(rows):
            if str(row[transaction_id_col - 1].value) == transaction_id_str:
                if not row[category_col - 1].value:
                    sheet.cells(i + 2, category_col).value = transaction.category
                    updated_count += 1
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(
                            f"Updated row {i + 2} with category: {transaction.category}"
                        )
                break

    logging.info(
        f"Batch update complete: {updated_count} transactions updated with categories"
    )
    return book
