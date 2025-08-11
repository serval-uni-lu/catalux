import yaml
import torch

from typing import List
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration class for model parameters and training settings."""
    
    # general Training
    num_classes: int = 2
    hidden_units: int = 128
    lr: float = 9e-3
    epochs: int = 1000
    weight_decay: float = 5e-4
    early_stopping_patience: int = 100
    num_folds: int = 4
    device: str = "auto"
    seed: int = 42
    dropout: float = 0.3
    
    # model-specific configs
    gat_heads: int = 1
    gat_dropout: float = 0.2
    gatv2_heads: int = 1
    gatv2_dropout: float = 0.2
    cheb_k: List[int] = None
    
    def __post_init__(self):
        """Set default for mutable fields and validate configuration."""
        if self.cheb_k is None:
            self.cheb_k = [1, 2]
        
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        if self.hidden_units <= 0:
            raise ValueError("Hidden units must be positive")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("Dropout must be between 0 and 1")
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file, using defaults for missing values."""
        config = cls()
        
        if Path(path).exists():
            with open(path, 'r') as f:
                yaml_data = yaml.safe_load(f) or {}
            
            for key, value in yaml_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    def get_device(self) -> torch.device:
        """Get torch device."""
        if self.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device)
        
        return device