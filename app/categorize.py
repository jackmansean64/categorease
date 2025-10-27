from typing import List, Tuple
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from pydantic import TypeAdapter
from toolkit.language_models.token_costs import calculate_total_prompt_cost, ModelName
import pandas as pd
from toolkit.language_models.model_connection import ChatModelsSetup
from models import Transaction, Category, CategorizedTransaction
from prompt_templates import analysis_template
from toolkit.language_models.parallel_processing import parallel_invoke_function
import logging
from bs4 import BeautifulSoup
import os
from mock_llm_response import MOCK_LLM_RESPONSE

logger = logging.getLogger(__name__)

TRANSACTION_HISTORY_LENGTH = 150
INVALID_CATEGORY = "Invalid"
UNKNOWN_CATEGORY = "Unknown"
MAX_TRANSACTIONS_TO_CATEGORIZE = 100

processed_transaction_ids = set()

def reset_categorization_session():
    """Reset the in-memory tracking of processed transactions for a new categorization session"""
    processed_transaction_ids.clear()
    logger.info(
        "Categorization session reset - all transactions can be reprocessed"
    )


def categorize_transaction_batch(
    transactions_data: list,
    categories_data: list,
    batch_number: int,
    batch_size: int
) -> dict:
    """Process a specific batch of transactions"""

    previously_categorized_transactions, uncategorized_transactions = (
        parse_transactions_data(transactions_data, categories_data)
    )

    unprocessed_transactions = [
        t for t in uncategorized_transactions if t.transaction_id not in processed_transaction_ids
    ]

    total_processed = len(processed_transaction_ids)

    remaining_limit = MAX_TRANSACTIONS_TO_CATEGORIZE - total_processed

    actual_batch_size = min(batch_size, remaining_limit, len(unprocessed_transactions))
    batch_transactions = unprocessed_transactions[:actual_batch_size]

    if not batch_transactions:
        logger.info(f"Batch {batch_number}: No unprocessed transactions remaining")
        reset_categorization_session()
        return {
            'categorized_transactions': [],
            'total_processed': total_processed,
            'completed': True
        }

    categories = parse_categories_data(categories_data)

    logger.info(
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

    except ValueError as e:
        logger.error(f"Batch {batch_number}: Data validation error: {e}")
        return {
            'error': str(e),
            'categorized_transactions': [],
            'total_processed': total_processed,
            'completed': True
        }
    except Exception as e:
        logger.error(f"Batch {batch_number}: Processing failed with error: {e}")
        raise e

    for transaction in batch_transactions:
        if transaction.transaction_id:
            processed_transaction_ids.add(transaction.transaction_id)

    total_cost = sum(cost for _, cost in categorized_transactions_and_costs)
    logger.info(f"Batch {batch_number} cost: ${total_cost:.4f}")

    categorized_transactions = [
        transaction for transaction, _ in categorized_transactions_and_costs
    ]

    new_total_processed = total_processed + len(batch_transactions)
    logger.info(f"Batch {batch_number}: Updated total_processed to {new_total_processed}")

    return {
        'categorized_transactions': [
            {
                'transaction_id': t.transaction_id,
                'category': t.category,
                'description': t.description,
                'date': t.date.isoformat() if hasattr(t.date, 'isoformat') else str(t.date)
            }
            for t in categorized_transactions if t.category != UNKNOWN_CATEGORY
        ],
        'total_processed': new_total_processed,
        'completed': new_total_processed >= MAX_TRANSACTIONS_TO_CATEGORIZE or not unprocessed_transactions[actual_batch_size:]
    }


def model_categorize_transaction(
    transaction: Transaction,
    categories: List[Category],
    categorized_transactions: List[Transaction],
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
            logger.info(f"Successfully categorized transaction {transaction_id} in {transaction_time:.1f}s after {attempt + 1} attempt(s)")
            return parsed_category, total_cost
        elif attempt == max_retries:
            parsed_category.category = UNKNOWN_CATEGORY
            transaction_time = time.time() - transaction_start_time
            logger.warning(
                f"Final attempt for transaction {transaction_id}, returning category: {parsed_category.category} after {transaction_time:.1f}s"
            )
            return parsed_category, total_cost
        else:
            logger.warning(
                f"Attempt {attempt + 1} failed for transaction {transaction_id}, retrying..."
            )

    transaction_time = time.time() - transaction_start_time
    logger.info(f"Completed transaction {transaction_id} in {transaction_time:.1f}s")
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

    use_mock_llm = os.getenv("USE_MOCK_LLM", "false").lower() == "true"

    if use_mock_llm:
        api_time = time.time() - api_start_time
        logger.info(f"MOCK LLM response for transaction {transaction_id} returned in {api_time:.1f}s")
        return MOCK_LLM_RESPONSE, 0.00

    prompt_template = PromptTemplate.from_template(analysis_template)

    formatted_prompt = prompt_template.format(
        categories=TypeAdapter(List[Category]).dump_python(categories),
        examples=TypeAdapter(List[Transaction]).dump_python(categorized_transactions),
        transaction=uncategorized_transaction.model_dump(),
    )

    analysis_response = chat_model.invoke(formatted_prompt)
    api_time = time.time() - api_start_time

    logger.info(f"LLM API call for transaction {transaction_id} completed in {api_time:.1f}s")

    total_cost = calculate_total_prompt_cost(
        analysis_response.response_metadata["usage"]["prompt_tokens"],
        analysis_response.response_metadata["usage"]["completion_tokens"],
        model_name,
    )

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
            logger.warning(
                f"No assigned_category tag found in analysis response for transaction {uncategorized_transaction.transaction_id}"
            )
            category = UNKNOWN_CATEGORY

    if category not in valid_categories:
        logger.warning(
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
    logger.info(f"Successfuly categorized transaction: {categorized_transaction}")

    return categorized_transaction


def parse_transactions_data(transactions_data: list, categories_data: list) -> Tuple[List[Transaction], List[Transaction]]:
    """Parse transaction data from plain arrays (no xlwings)"""
    transaction_columns = [
        "Date",
        "Description",
        "Category",
        "Amount",
        "Account",
        "Transaction ID",
    ]

    if not transactions_data or len(transactions_data) < 2:
        return [], []

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


def parse_categories_data(categories_data: list) -> List[Category]:
    """Parse category data from plain arrays (no xlwings)"""
    category_columns = ["Category", "Group", "Type"]

    if not categories_data or len(categories_data) < 2:
        return []

    categories_df = pd.DataFrame(categories_data[1:], columns=categories_data[0])
    for col in category_columns:
        if col not in categories_df.columns:
            categories_df[col] = None
    categories_df = categories_df[category_columns]
    return _convert_df_to_categories(categories_df)


def clean_amount(amount):
    """Clean and validate amount values, raising descriptive errors for invalid data"""
    if pd.isna(amount) or amount is None:
        return None

    if isinstance(amount, (int, float)):
        return float(amount)

    if isinstance(amount, str):
        if not amount.strip():
            return None

        try:
            cleaned = amount.replace("$", "").replace(",", "").strip()
            if not cleaned:
                return None
            return float(cleaned)
        except ValueError:
            raise ValueError(
                f"Invalid amount value: '{amount}'. Amount must be a number (e.g., '100', '$100.50', '1,234.56')"
            )

    raise ValueError(
        f"Invalid amount type: '{amount}' (type: {type(amount).__name__}). Amount must be a number or numeric string"
    )


def _convert_df_to_transactions(df: pd.DataFrame) -> List[Transaction]:
    df = df.copy()

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
            row_dict = row.to_dict()
            if 'Amount' in row_dict:
                row_dict['Amount'] = clean_amount(row_dict['Amount'])

            transaction = Transaction(**row_dict)
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


