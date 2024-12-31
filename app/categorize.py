from xlwings import Book
import pandas as pd

def read_excel_data(book: Book) -> tuple[pd.DataFrame, pd.DataFrame]:
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
