import torch
import numpy as np
import pandas as pd
from functools import cached_property


class Transaction:
    """Represents an Ethereum transaction."""
    
    def __init__(self, from_id: int, to_id: int, value: float, 
                 gas: float = 21000, gas_price: float = 30e9):
        self.from_id = from_id
        self.to_id = to_id
        self.value = value
        self.gas = gas
        self.gas_price = gas_price
        self.gas_used = gas * np.random.uniform(0.9, 1.0)

    @cached_property
    def gas_cost(self) -> float:
        """Calculate gas cost in Wei."""
        return self.gas_used * self.gas_price

    @cached_property
    def total_cost(self) -> float:
        """Calculate total cost for sender."""
        return self.value + self.gas_cost
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
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
        """Convert to pandas DataFrame."""
        return pd.DataFrame([self.to_dict()])
    
    def to_edge_features(self, device: torch.device = None) -> torch.Tensor:
        """Convert to edge feature tensor with device support."""
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