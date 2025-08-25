from typing import List
from .transaction import Transaction


class Account:
    """Represents an Ethereum account with transaction history."""
    
    def __init__(self, node_id: int, balance: float = 0.0) -> None:
        """Initialize account with node ID and balance.
        
        Args:
            node_id: Unique identifier for the account
            balance: Initial account balance in Wei
        """
        self.node_id = node_id
        self.balance = balance
        self.transaction_history: List[Transaction] = []
        
    def can_afford(self, total_cost: float) -> bool:
        """Check if account can afford transaction including gas costs.
        
        Args:
            total_cost: Total cost including transaction value and gas
            
        Returns:
            True if account has sufficient balance
        """
        return self.balance >= total_cost
    
    def send(self, tx: Transaction) -> bool:
        """Send a transaction if funds are available.
        
        Args:
            tx: Transaction object to send
            
        Returns:
            True if transaction was successfully sent
        """
        if not self.can_afford(tx.total_cost):
            return False
        self.balance -= tx.total_cost
        self.transaction_history.append(tx)
        return True

    def receive(self, tx: Transaction) -> None:
        """Receive a transaction and update balance.
        
        Args:
            tx: Transaction object received
        """
        self.balance += tx.value
        self.transaction_history.append(tx)
    
    def get_transaction_count(self) -> int:
        """Get total number of transactions for this account.
        
        Returns:
            Total transaction count
        """
        return len(self.transaction_history)
    
    def get_total_sent(self) -> float:
        """Get total value sent by this account.
        
        Returns:
            Sum of all outgoing transaction values
        """
        return sum(tx.value for tx in self.transaction_history if tx.from_id == self.node_id)
    
    def get_total_received(self) -> float:
        """Get total value received by this account.
        
        Returns:
            Sum of all incoming transaction values
        """
        return sum(tx.value for tx in self.transaction_history if tx.to_id == self.node_id)
    
    def __repr__(self) -> str:
        return f"Account(id={self.node_id}, balance={self.balance:.2e}, txs={len(self.transaction_history)})"