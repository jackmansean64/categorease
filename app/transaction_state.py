from typing import Dict, List, Set
from models import CategorizedTransaction, Transaction


class TransactionStateManager:
    """
    Manages the state of transactions between API calls.
    Tracks which transactions have been categorized and which are still pending.
    """

    def __init__(self):
        self._categorized_transactions: Dict[str, CategorizedTransaction] = {}
        self._in_progress_transaction_ids: Set[str] = set()

    def get_uncategorized_transactions(
        self, all_transactions: List[Transaction]
    ) -> List[Transaction]:
        """
        Filter the list of all transactions to only include those that haven't been
        categorized yet and aren't currently being processed.
        """
        to_process = []
        for transaction in all_transactions:
            if not transaction.category and transaction.transaction_id:
                # Skip if already categorized by our system or currently in progress
                if (
                    transaction.transaction_id not in self._categorized_transactions
                    and transaction.transaction_id
                    not in self._in_progress_transaction_ids
                ):
                    to_process.append(transaction)
        return to_process

    def mark_transactions_in_progress(self, transactions: List[Transaction]):
        """Mark transactions as currently being processed."""
        for transaction in transactions:
            if transaction.transaction_id:
                self._in_progress_transaction_ids.add(transaction.transaction_id)

    def add_categorized_transactions(self, categorized: List[CategorizedTransaction]):
        """Add newly categorized transactions to the state."""
        for transaction in categorized:
            self._categorized_transactions[transaction.transaction_id] = transaction
            # Remove from in-progress set if present
            self._in_progress_transaction_ids.discard(transaction.transaction_id)

    def get_all_categorized_transactions(self) -> List[CategorizedTransaction]:
        """Get all categorized transactions."""
        return list(self._categorized_transactions.values())

    def reset(self):
        """Clear all state."""
        self._categorized_transactions.clear()
        self._in_progress_transaction_ids.clear()


# Singleton instance to be used across API calls
transaction_state = TransactionStateManager()
