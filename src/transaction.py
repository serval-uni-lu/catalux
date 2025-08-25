import torch
import numpy as np
import pandas as pd
from functools import cached_property
from typing import Dict, Any, Optional


class Transaction:
    """Represents an Ethereum transaction with gas cost calculations."""
    
    def __init__(self, from_id: int, to_id: int, value: float, 
                 gas: float = 21000, gas_price: float = 30e9) -> None:
        """Initialize Ethereum transaction.
        
        Args:
            from_id: Sender account ID
            to_id: Recipient account ID
            value: Transaction value in Wei
            gas: Gas limit for the transaction
            gas_price: Gas price in Wei per gas unit
        """
        self.from_id = from_id
        self.to_id = to_id
        self.value = value
        self.gas = gas
        self.gas_price = gas_price
        self.gas_used = gas * np.random.uniform(0.9, 1.0)

    @cached_property
    def gas_cost(self) -> float:
        """Calculate total gas cost in Wei.
        
        Returns:
            Gas cost calculated as gas_used * gas_price
        """
        return self.gas_used * self.gas_price

    @cached_property
    def total_cost(self) -> float:
        """Calculate total cost for the sender including gas.
        
        Returns:
            Sum of transaction value and gas cost
        """
        return self.value + self.gas_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary format.
        
        Returns:
            Dictionary with transaction attributes
        """
        return {
            'from_id': self.from_id,
            'to_id': self.to_id,
            'value': self.value,
            'gas': self.gas,
            'gas_price': self.gas_price,
            'input': '0x',
            'receipt_gas_used': self.gas_used
        }
        
    def to_df(self) -> pd.DataFrame:
        """Convert transaction to pandas DataFrame.
        
        Returns:
            Single-row DataFrame containing transaction data
        """
        return pd.DataFrame([self.to_dict()])
    
    def to_edge_features(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert transaction to edge feature tensor for graph networks.
        
        Args:
            device: Target device for the tensor
            
        Returns:
            Feature tensor with transaction attributes
        """
        features = torch.tensor([
            self.value,
            self.gas,
            self.gas_price,
            self.gas_used,
            self.gas_cost,
            self.gas_used / max(self.gas, 1e-8)
        ], dtype=torch.float32)
        
        if device is not None:
            features = features.to(device)
            
        return features