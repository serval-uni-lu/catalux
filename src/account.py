from typing import List
from .transaction import Transaction

class Account:
    """Represents an Ethereum account with transaction history."""
    
    def __init__(self, node_id: int, balance: float = 0.0):
        self.node_id = node_id
        self.balance = balance
        self.transaction_history: List[Transaction] = []
        
    def can_afford(self, total_cost: float) -> bool:
        """Check if account can afford transaction + gas."""
        return self.balance >= total_cost
    
    def send(self, tx: Transaction) -> bool:
        """Send a transaction if funds are available."""
        if not self.can_afford(tx.total_cost):
            return False
        self.balance -= tx.total_cost
        self.transaction_history.append(tx)
        return True

    def receive(self, tx: Transaction) -> None:
        """Receive a transaction."""
        self.balance += tx.value
        self.transaction_history.append(tx)
    
    def get_transaction_count(self) -> int:
        """Get total number of transactions."""
        return len(self.transaction_history)
    
    def get_total_sent(self) -> float:
        """Get total value sent."""
        return sum(tx.value for tx in self.transaction_history if tx.from_id == self.node_id)
    
    def get_total_received(self) -> float:
        """Get total value received."""
        return sum(tx.value for tx in self.transaction_history if tx.to_id == self.node_id)
    
    def __repr__(self) -> str:
        return f"Account(id={self.node_id}, balance={self.balance:.2e}, txs={len(self.transaction_history)})"