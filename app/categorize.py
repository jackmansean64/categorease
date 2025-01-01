from typing import List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import TypeAdapter
from toolkit.language_models.token_costs import calculate_total_prompt_cost, ModelName
from xlwings import Book
import pandas as pd
from toolkit.language_models.model_connection import ChatModelsSetup
from app.models import Transaction, Category, CategorizedTransaction
from app.prompt_templates import analysis_template, serialize_categories_template
from toolkit.language_models.parallel_processing import parallel_invoke_function


def categorize_transactions_in_book(book: Book) -> Book:
    transactions_df, categories_df = _read_excel_data(book)
    transactions = _convert_df_to_transactions(transactions_df)
    categories = _convert_df_to_categories(categories_df)

    categorized_transactions = [transaction for transaction in transactions if transaction.category]
    uncategorized_transactions = [transaction for transaction in transactions if not transaction.category]

    print(f"Uncategorized Transaction Total: {len(uncategorized_transactions)}")

    categorized_transactions: List[CategorizedTransaction] = parallel_invoke_function(
        function=process_transaction,
        variable_args=uncategorized_transactions,
        categories=categories,
        categorized_transactions=categorized_transactions,
    )

    return update_categories_in_sheet(book, categorized_transactions)


def process_transaction(transaction, categories, categorized_transactions):
    chat_models = ChatModelsSetup()

    analysis_prompt_template = PromptTemplate.from_template(analysis_template)
    serialization_prompt_template = PromptTemplate.from_template(serialize_categories_template)

    formatted_analysis_prompt = analysis_prompt_template.format(
        categories=TypeAdapter(List[Category]).dump_python(categories),
        examples=TypeAdapter(List[Transaction]).dump_python(categorized_transactions[:200]),
        transaction=transaction.model_dump(),
    )

    analysis_response = chat_models.claude_35_v2_sonnet_chat.invoke(formatted_analysis_prompt)

    formatted_serialization_prompt = serialization_prompt_template.format(
        transaction=transaction,
        json_structure=CategorizedTransaction.model_json_schema()
    )

    serialization_prompt_with_analysis_response = [
        AIMessage(content=analysis_response.content),
        HumanMessage(content=formatted_serialization_prompt),
    ]

    return CategorizedTransaction.model_validate_json(
        chat_models.claude_35_haiku_chat.invoke(serialization_prompt_with_analysis_response).content
    )


def analyze_transaction(
        uncategorized_transaction: Transaction,
        categories: List[Category],
        categorized_transactions: List[Transaction]
) -> str:
    TRANSACTION_HISTORY_LENGTH = 200
    prompt_template = PromptTemplate.from_template(analysis_template)

    formatted_prompt = prompt_template.format(
        categories=TypeAdapter(List[Category]).dump_python(categories),
        examples=TypeAdapter(List[Transaction]).dump_python(categorized_transactions[:TRANSACTION_HISTORY_LENGTH]),
        transaction=uncategorized_transaction.model_dump(),
    )
    # print(formatted_prompt)

    chat_models = ChatModelsSetup()
    analysis_response = chat_models.claude_35_haiku_chat.invoke(formatted_prompt)

    total_cost = calculate_total_prompt_cost(
        analysis_response.response_metadata["usage"]["prompt_tokens"],
        analysis_response.response_metadata["usage"]["completion_tokens"],
        ModelName.HAIKU_3_5
    )
    print(f"Total Cost: ${total_cost}")

    return analysis_response.content


def parse_category_from_analysis(uncategorized_transaction: Transaction, analysis_response: str) -> Category:
    serialization_prompt_template = PromptTemplate.from_template(serialize_categories_template)

    formatted_prompt = serialization_prompt_template.format(
        transaction=uncategorized_transaction,
        json_structure=CategorizedTransaction.model_json_schema()
    )

    prompt = [
        AIMessage(content=analysis_response),
        HumanMessage(content=formatted_prompt),
    ]

    chat_models = ChatModelsSetup()
    category_assignment_response = chat_models.claude_35_haiku_chat.invoke(prompt)
    assigned_category = CategorizedTransaction.model_validate_json(category_assignment_response.content)
    # print(assigned_category)

    total_cost = calculate_total_prompt_cost(
        category_assignment_response.response_metadata["usage"]["prompt_tokens"],
        category_assignment_response.response_metadata["usage"]["completion_tokens"],
        ModelName.HAIKU_3_5
    )
    print(f"Total Cost: ${total_cost}")

    return assigned_category


def _read_excel_data(book: Book) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "Date Added"
    ]

    category_columns = [
        "Category",
        "Group",
        "Type"
    ]

    transactions_sheet = book.sheets['Transactions']
    transactions_data = transactions_sheet.range('A1').expand().value
    transactions_df = pd.DataFrame(transactions_data[1:], columns=transactions_data[0])

    transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])
    transactions_df['Date Added'] = pd.to_datetime(transactions_df['Date Added'])
    transactions_df = transactions_df.astype(
        {
            'Account #': str,
            'Transaction ID': str,
            'Account ID': str,
            'Check Number': str
        }
    )
    for col in transaction_columns:
        if col not in transactions_df.columns:
            transactions_df[col] = None
    transactions_df = transactions_df[transaction_columns]

    categories_sheet = book.sheets['Categories']
    categories_data = categories_sheet.range('A1').expand().value
    categories_df = pd.DataFrame(categories_data[1:], columns=categories_data[0])
    for col in category_columns:
        if col not in categories_df.columns:
            categories_df[col] = None
    categories_df = categories_df[category_columns]

    return transactions_df, categories_df


def clean_amount(amount):
    if pd.isna(amount):
        return None
    if isinstance(amount, str):
        # Remove currency symbol and commas
        cleaned = amount.replace('$', '').replace(',', '').strip()
        return float(cleaned)
    return float(amount)


def _convert_df_to_transactions(df: pd.DataFrame) -> List[Transaction]:
    df = df.copy()

    # Clean amount column first
    df['Amount'] = df['Amount'].apply(clean_amount)

    # Clean all optional columns
    columns_to_clean = [
        'Category', 'Labels', 'Notes', 'Check Number',
        'Account', 'Account #', 'Institution', 'Month',
        'Week', 'Transaction ID', 'Account ID', 'Full Description'
    ]
    for col in columns_to_clean:
        df[col] = df[col].where(pd.notna(df[col]), None)

    transactions = []
    for index, row in df.iterrows():
        try:
            transaction = Transaction(**row.to_dict())
            transactions.append(transaction)
        except Exception as e:
            raise ValueError(f"Row {index} failed validation: {str(e)}\nData: {row.to_dict()}")

    return transactions


def _convert_df_to_categories(df: pd.DataFrame) -> List[Category]:
    df = df.copy()

    # Clean optional columns
    columns_to_clean = ['Group', 'Type']
    for col in columns_to_clean:
        df[col] = df[col].where(pd.notna(df[col]), None)

    categories = []
    for index, row in df.iterrows():
        try:
            category = Category(**row.to_dict())
            categories.append(category)
        except Exception as e:
            raise ValueError(f"Row {index} failed validation: {str(e)}\nData: {row.to_dict()}")

    return categories


def update_categories_in_sheet(book: Book, categorized_transactions: List[CategorizedTransaction]) -> Book:
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
                if not row[category_col - 1].value:
                    sheet.cells(i + 2, category_col).value = transaction.category
                break

    return book
