import yaml
import torch

from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration class."""
    
    # general training
    num_classes: int = 2
    lr: float = 5e-3
    epochs: int = 250
    dropout: float = 0.2
    weight_decay: float = 5e-4
    early_stopping_patience: int = 30
    device: str = "auto"
    seed: int = 42
    
    # architecture parameters
    layers: int = 2
    hidden_units: int = 256
    use_batch_norm: bool = False

    # model-specific configs
    gat_heads: int = 4
    gat_dropout: float = 0.2
    gatv2_heads: int = 4
    gatv2_dropout: float = 0.2
    cheb_k: list[int] = [2, 3]
    
    # additional optimization parameters
    gradient_clipping: float = 1.0
    scheduler_patience: int = 10
    scheduler_factor: float = 0.7
    
    # model selection criteria
    primary_metric: str = "f1"
    secondary_metric: str = "recall"
    min_recall_threshold: float = 0.80
    
    # tabular model parameters
    # RealMLP
    realmlp_lr: float = 0.001
    realmlp_layers: int = 4
    realmlp_hidden: int = 512
    realmlp_dropout: float = 0.25
    realmlp_batch_size: int = 256
    realmlp_weight_decay: float = 0.00001
    
    # TabM
    tabm_blocks: int = 3
    tabm_d_block: int = 384
    tabm_k: int = 16
    tabm_dropout: float = 0.12
    tabm_lr: float = 0.002
    tabm_batch_size: int = 256
    tabm_weight_decay: float = 0.0001
    
    def __post_init__(self) -> None:
        """Set default for mutable fields and validate configuration."""            
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # validation
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        if self.hidden_units <= 0:
            raise ValueError("Hidden units must be positive")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("Dropout must be between 0 and 1")
        if self.layers < 1:
            raise ValueError("Number of layers must be at least 1")
        if self.gat_heads < 1:
            raise ValueError("Number of GAT heads must be at least 1")
        if self.min_recall_threshold < 0 or self.min_recall_threshold > 1:
            raise ValueError("Minimum recall threshold must be between 0 and 1")
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        config = cls()
        
        if Path(path).exists():
            with open(path, 'r') as f:
                yaml_data = yaml.safe_load(f) or {}
            
            for key, value in yaml_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    print(f"Warning: Unknown parameter '{key}' in config file")
        
        return config
    
    def get_device(self) -> torch.device:
        """Get torch device."""
        if self.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device)
        
        return device
    
    def get_model_specific_params(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific parameters.
        
        Args:
            model_name: Name of the model to get parameters for
            
        Returns:
            Dictionary containing model-specific configuration parameters
        """
        base_params = {
            'hidden_units': self.hidden_units,
            'dropout': self.dropout,
            'layers': self.layers,
            'use_batch_norm': self.use_batch_norm
        }
        
        if model_name in ['GAT']:
            base_params.update({
                'gat_heads': self.gat_heads,
                'gat_dropout': self.gat_dropout
            })
        elif model_name in ['GATv2']:
            base_params.update({
                'gatv2_heads': self.gatv2_heads,
                'gatv2_dropout': self.gatv2_dropout
            })
        elif model_name in ['Chebyshev']:
            base_params.update({
                'cheb_k': self.cheb_k
            })
        
        return base_params