from typing import List, Tuple, Optional
from pydantic import TypeAdapter
from toolkit.language_models.token_costs import calculate_total_prompt_cost, ModelName
from xlwings import Book
import pandas as pd
from models import Transaction, Category, CategorizedTransaction
from prompt_templates import analysis_template, serialize_categories_template
from toolkit.language_models.parallel_processing import parallel_invoke_function
import logging
from bs4 import BeautifulSoup
import os
import threading
import time

TRANSACTION_HISTORY_LENGTH = 150
INVALID_CATEGORY = "Invalid"
UNKNOWN_CATEGORY = "Unknown"
MAX_TRANSACTIONS_TO_CATEGORIZE = 100

processed_transaction_ids = set()

# Will be set by server_flask.py
_xlwings_lock: Optional[threading.Lock] = None

def set_xlwings_lock(lock: threading.Lock):
    """Set the global xlwings lock from the Flask app"""
    global _xlwings_lock
    _xlwings_lock = lock

def reset_categorization_session():
    """Reset the in-memory tracking of processed transactions for a new categorization session"""
    processed_transaction_ids.clear()
    logging.info(
        "Categorization session reset - all transactions can be reprocessed"
    )


def categorize_transaction_batch(
    book: Book, batch_number: int, batch_size: int
) -> Book:
    """Process a specific batch of transactions"""
    import time
    batch_start = time.time()

    logging.info(f"[TIMING] Batch {batch_number} started")
    previously_categorized_transactions, uncategorized_transactions = (
        retrieve_transactions(book)
    )
    logging.info(f"[TIMING] Batch {batch_number}: Transactions retrieved at {time.time() - batch_start:.3f}s")

    unprocessed_transactions = [
        t for t in uncategorized_transactions if t.transaction_id not in processed_transaction_ids
    ]

    # Lock batch_info sheet read
    try:
        lock_start = time.time()
        with _xlwings_lock:
            lock_wait = time.time() - lock_start
            if lock_wait > 0.1:
                logging.warning(f"[LOCK] batch_info read waited {lock_wait:.3f}s for xlwings lock")

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
    logging.info(f"[TIMING] Batch {batch_number}: Categories retrieved at {time.time() - batch_start:.3f}s")

    logging.info(
        f"Processing batch {batch_number}: processing {len(batch_transactions)} transactions ({len(unprocessed_transactions)} unprocessed from {len(uncategorized_transactions)} total uncategorized)"
    )

    try:
        disable_multi_threading = os.getenv("DISABLE_MULTI_THREADING", "false").lower() == "true"

        categorization_start = time.time()
        if disable_multi_threading:
            categorized_transactions_and_costs = [
                model_categorize_transaction(
                    transaction=transaction,
                    categories=categories,
                    categorized_transactions=previously_categorized_transactions[:TRANSACTION_HISTORY_LENGTH],
                )
                for transaction in batch_transactions
            ]
        else:
            categorized_transactions_and_costs = parallel_invoke_function(
                function=model_categorize_transaction,
                variable_args=batch_transactions,
                categories=categories,
                categorized_transactions=previously_categorized_transactions[:TRANSACTION_HISTORY_LENGTH],
            )
        logging.info(f"[TIMING] Batch {batch_number}: Categorization complete at {time.time() - batch_start:.3f}s (took {time.time() - categorization_start:.3f}s)")

    except Exception as e:
        logging.error(f"Batch {batch_number}: Processing failed with error: {e}")
        raise e

    for transaction in batch_transactions:
        if transaction.transaction_id:
            processed_transaction_ids.add(transaction.transaction_id)

    total_cost = sum(cost for _, cost in categorized_transactions_and_costs)
    logging.info(f"Batch {batch_number} cost: ${total_cost:.4f}")

    categorized_transactions = [
        transaction for transaction, _ in categorized_transactions_and_costs
    ]

    # Lock batch_info sheet write
    try:
        lock_start = time.time()
        with _xlwings_lock:
            lock_wait = time.time() - lock_start
            if lock_wait > 0.1:
                logging.warning(f"[LOCK] batch_info write waited {lock_wait:.3f}s for xlwings lock")

            batch_info_sheet = book.sheets["_batch_info"]
            new_total_processed = total_processed + len(batch_transactions)
            batch_info_sheet.range("B4").value = new_total_processed
        logging.info(f"Batch {batch_number}: Updated total_processed to {new_total_processed}")
    except Exception as e:
        logging.warning(f"Failed to update total_processed count: {e}")

    logging.info(f"[TIMING] Batch {batch_number}: Calling update_categories_in_sheet_batch at {time.time() - batch_start:.3f}s")
    result = update_categories_in_sheet_batch(
        book, categorized_transactions, uncategorized_transactions
    )
    logging.info(f"[TIMING] Batch {batch_number} complete at {time.time() - batch_start:.3f}s")
    return result


def model_categorize_transaction(
    transaction: Transaction,
    categories: List[Category],
    categorized_transactions: List[Transaction],
) -> Tuple[CategorizedTransaction, float]:
    import time
    
    transaction_start_time = time.time()
    transaction_id = getattr(transaction, 'transaction_id', 'unknown')
    
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
            logging.info(f"Successfully categorized transaction {transaction_id} in {transaction_time:.1f}s after {attempt + 1} attempt(s)")
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
    model_name: ModelName,
) -> Tuple[str, float]:
    import time

    api_start_time = time.time()
    transaction_id = getattr(uncategorized_transaction, 'transaction_id', 'unknown')

    # Static response for debugging timeout issues
    static_response = """Let's solve this step by step:

1. Transaction Details:
- Description: "Taylor swift toronto on"
- Amount: -$73.68 (negative, so it's an expense)
- Account: Scotia Momentum VISA Infinite

2. Analyzing the Description:
- "Taylor swift" suggests this is related to a concert or entertainment event
- "toronto on" indicates the location of the event

3. Reviewing Existing Categories:
- I see relevant categories in the Entertainment group:
  - "Entertainment"
  - "Concerts and Shows"

4. Examining Past Transactions:
- I see other entertainment-related transactions like concerts and shows
- The negative amount and event-related description strongly suggest this is a concert expense

5. Confidence Assessment:
- The description clearly indicates a concert
- The category "Concerts and Shows" is a perfect match
- I am >90% confident in this categorization

6. Reasoning:
- The transaction is a negative expense for a Taylor Swift concert
- "Concerts and Shows" is the most precise and appropriate category

<assigned_category>Concerts and Shows</assigned_category>"""

    api_time = time.time() - api_start_time

    logging.info(f"STATIC RESPONSE for transaction {transaction_id} returned in {api_time:.1f}s")

    total_cost = 0.00

    return static_response, total_cost


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

    # Lock xlwings operations to prevent concurrent access deadlock
    lock_start = time.time()
    with _xlwings_lock:
        lock_wait = time.time() - lock_start
        if lock_wait > 0.1:
            logging.warning(f"[LOCK] retrieve_transactions waited {lock_wait:.3f}s for xlwings lock")

        logging.debug("[LOCK] retrieve_transactions acquired xlwings lock")
        transactions_sheet = book.sheets["Transactions"]
        transactions_data = transactions_sheet.range("A1").expand().value
        logging.debug("[LOCK] retrieve_transactions releasing xlwings lock")

    # Process data outside the lock (parallel-safe)
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

    # Lock xlwings operations to prevent concurrent access deadlock
    lock_start = time.time()
    with _xlwings_lock:
        lock_wait = time.time() - lock_start
        if lock_wait > 0.1:
            logging.warning(f"[LOCK] retrieve_categories waited {lock_wait:.3f}s for xlwings lock")

        logging.debug("[LOCK] retrieve_categories acquired xlwings lock")
        categories_sheet = book.sheets["Categories"]
        categories_data = categories_sheet.range("A1").expand().value
        logging.debug("[LOCK] retrieve_categories releasing xlwings lock")

    # Process data outside the lock (parallel-safe)
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
    update_start = time.time()

    if not categorized_transactions:
        return book

    # Prepare lookup dictionary outside the lock (parallel-safe)
    transaction_id_to_category = {
        str(transaction.transaction_id): transaction.category
        for transaction in categorized_transactions
        if transaction.transaction_id is not None
        and transaction.category != UNKNOWN_CATEGORY
    }

    # Lock xlwings operations to prevent concurrent access deadlock
    lock_start = time.time()
    with _xlwings_lock:
        lock_wait = time.time() - lock_start
        if lock_wait > 0.1:
            logging.warning(f"[LOCK] update_categories_in_sheet_batch waited {lock_wait:.3f}s for xlwings lock")

        logging.debug("[LOCK] update_categories_in_sheet_batch acquired xlwings lock")
        logging.info(f"[TIMING] Sheet update: Getting sheet and headers")
        sheet = book.sheets["Transactions"]
        headers = sheet.range("A1").expand("right").value
        transaction_id_col = headers.index("Transaction ID") + 1
        category_col = headers.index("Category") + 1
        logging.info(f"[TIMING] Sheet update: Headers retrieved at {time.time() - update_start:.3f}s")

        logging.info(f"[TIMING] Sheet update: Getting rows")
        # Read all row values into memory at once to avoid thousands of xlwings calls
        rows_range = sheet.tables[0].data_body_range
        rows_values = rows_range.value  # Read all data in one xlwings operation
        logging.info(f"[TIMING] Sheet update: Rows retrieved at {time.time() - update_start:.3f}s")
        logging.info(f"[TIMING] Sheet update: Total rows in sheet: {len(rows_values) if rows_values else 0}")

        updated_count = 0
        logging.info(f"[TIMING] Sheet update: Starting row updates")

        # Collect all updates first (now using Python data, no xlwings operations)
        updates = []  # List of (row_index, category) tuples
        for transaction in categorized_transactions:
            if transaction.transaction_id is None:
                logging.warning(
                    f"No transaction ID present for {transaction.description}, category can't be assigned."
                )
                continue

            transaction_id_str = str(transaction.transaction_id)
            if transaction_id_str not in transaction_id_to_category:
                continue

            # Loop through Python list, not xlwings Range objects
            if rows_values:
                for i, row_data in enumerate(rows_values):
                    # row_data is now a Python list, not an xlwings Range
                    if str(row_data[transaction_id_col - 1]) == transaction_id_str:
                        if not row_data[category_col - 1]:
                            updates.append((i + 2, transaction.category))
                            if logging.getLogger().isEnabledFor(logging.DEBUG):
                                logging.debug(
                                    f"Will update row {i + 2} with category: {transaction.category}"
                                )
                        break

        # Apply all updates with logging to identify hangs
        logging.info(f"[TIMING] Sheet update: Collected {len(updates)} updates to apply")
        if updates:
            for idx, (row_idx, category) in enumerate(updates):
                try:
                    logging.info(f"[TIMING] Sheet update: Writing row {row_idx} ({idx+1}/{len(updates)})")
                    sheet.cells(row_idx, category_col).value = category
                    updated_count += 1
                    logging.info(f"[TIMING] Sheet update: Row {row_idx} written successfully")
                except Exception as e:
                    logging.error(f"Failed to update row {row_idx}: {e}")
                    continue

        logging.info(f"[TIMING] Sheet update: Updates applied at {time.time() - update_start:.3f}s")
        logging.debug("[LOCK] update_categories_in_sheet_batch releasing xlwings lock")

    logging.info(
        f"Batch update complete: {updated_count} transactions updated with categories"
    )
    logging.info(f"[TIMING] Sheet update: Complete at {time.time() - update_start:.3f}s")
    return book
