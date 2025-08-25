import torch
import warnings
import numpy as np
from typing import Dict
from pathlib import Path
from torch.nn import Module

import tabm
import lightgbm as lgb
from rtdl_revisiting_models import MLP as RealMLP_Base

from .config import Config


class RealMLP(Module):
    """Wrapper for RTDL MLP model for binary classification."""
    
    def __init__(self, num_features: int, hidden_units: int = 256, num_classes: int = 2,
                 dropout: float = 0.1, num_layers: int = 3):
        super().__init__()
        
        if RealMLP_Base is None:
            raise ImportError("rtdl_revisiting_models not installed. Install with: pip install rtdl_revisiting_models")
        
        self.num_classes = num_classes
        out_dim = 1 if num_classes == 2 else num_classes
        
        self.model = RealMLP_Base(
            d_in=num_features,
            d_out=out_dim,
            n_blocks=num_layers,
            d_block=hidden_units,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        
        if self.num_classes == 2:
            probs = torch.sigmoid(logits)
            neg_probs = 1 - probs
            return torch.log(torch.cat([neg_probs, probs], dim=-1) + 1e-10)
        else:
            return torch.log_softmax(logits, dim=-1)


class TabMWrapper(Module):
    """Wrapper for TabM model for binary classification."""
    
    def __init__(self, num_features: int, num_classes: int, 
                 n_blocks: int = 3, d_block: int = 256, k: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        
        if tabm is None:
            raise ImportError("tabm not installed. Install with: pip install tabm")
        
        self.k = k
        self.num_classes = num_classes
        
        self.model = tabm.TabM(
            n_num_features=num_features,
            cat_cardinalities=[],  # No categorical features
            d_out=1 if num_classes == 2 else num_classes,
            n_blocks=n_blocks,
            d_block=d_block,
            k=k,  # number of MLP backbones (ensemble size)
            dropout=dropout,
            start_scaling_init='random-signs'
        )
    
    def forward(self, x: torch.Tensor, return_ensemble: bool = False) -> torch.Tensor:
        """Forward pass for TabM.
        
        Args:
            x: Input features
            return_ensemble: If True, return all k predictions (for training)
                           If False, return averaged predictions (for inference)
        """
        logits = self.model(x, None)
        
        if self.num_classes == 2:
            if return_ensemble:
                probs = torch.sigmoid(logits)
                neg_probs = 1 - probs
                return torch.cat([neg_probs, probs], dim=-1)
            else:
                # average across ensemble for inference
                probs = torch.sigmoid(logits).mean(dim=1)  # (batch_size, 1)
                neg_probs = 1 - probs
                return torch.log(torch.cat([neg_probs, probs], dim=-1) + 1e-10)
        else:
            if return_ensemble:
                return logits
            else:
                return logits.mean(dim=1)


class LightGBMWrapper:
    """Wrapper for LightGBM classifier to match PyTorch interface."""
    
    def __init__(self, num_features: int, num_classes: int, config: Config):
        if lgb is None:
            raise ImportError("lightgbm not installed. Install with: pip install lightgbm")
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.config = config
        self.model = None
        self.device = config.get_device()
        
        # lightGBM parameters
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary' if num_classes == 2 else 'multiclass',
            'num_class': 1 if num_classes == 2 else num_classes,
            'metric': 'binary_logloss' if num_classes == 2 else 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': -1,
            'lambda_l1': 0.0,
            'lambda_l2': 0.0,
            'min_child_samples': 20,
            'max_depth': -1
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              num_boost_round: int = 1000, early_stopping_rounds: int = 50):
        """Train LightGBM model."""
        
        # create lightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions."""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # use best_iteration if available, otherwise use all iterations
        num_iteration = getattr(self.model, 'best_iteration', None)
        predictions = self.model.predict(X, num_iteration=num_iteration)
        
        if self.num_classes == 2:
            return (predictions > 0.5).astype(int)
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # use best_iteration if available, otherwise use all iterations
        num_iteration = getattr(self.model, 'best_iteration', None)
        predictions = self.model.predict(X, num_iteration=num_iteration)
        
        if self.num_classes == 2:
            prob_positive = predictions
            prob_negative = 1 - predictions
            return np.column_stack([prob_negative, prob_positive])
        else:
            return predictions
    
    def save(self, path: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save_model(path)
    
    def load(self, path: str):
        """Load model from file."""
        self.model = lgb.Booster(model_file=path)
        return self
    
    def to(self, device):
        """Compatibility method for PyTorch interface."""
        return self
    
    def eval(self):
        """Compatibility method for PyTorch interface."""
        return self
    
    def train_mode(self):
        """Compatibility method for PyTorch interface."""
        return self


def get_tabular_model_constructors(num_features: int, config: Config) -> Dict[str, callable]:
    """Get constructors for tabular models."""
    
    constructors = {}
    
    # RealMLP
    if RealMLP_Base is not None:
        constructors['RealMLP'] = lambda: RealMLP(
            num_features=num_features,
            hidden_units=getattr(config, 'realmlp_hidden', 256),
            num_classes=config.num_classes,
            dropout=getattr(config, 'realmlp_dropout', 0.1),
            num_layers=getattr(config, 'realmlp_layers', 3)
        )
    
    # TabM
    if tabm is not None:
        constructors['TabM'] = lambda: TabMWrapper(
            num_features=num_features,
            num_classes=config.num_classes,
            n_blocks=getattr(config, 'tabm_blocks', 3),
            d_block=getattr(config, 'tabm_d_block', 256),
            k=getattr(config, 'tabm_k', 8),
            dropout=getattr(config, 'tabm_dropout', 0.0)
        )
    
    # lightGBM
    if lgb is not None:
        constructors['LightGBM'] = lambda: LightGBMWrapper(
            num_features=num_features,
            num_classes=config.num_classes,
            config=config
        )
    
    return constructors


def load_pretrained_tabular(model_path: str, config: Config, num_features: int = None):
    """Load a pretrained tabular model.
    
    Args:
        model_path: Path to saved model
        config: Configuration object
        num_features: Number of input features (auto-detected if None)
        
    Returns:
        Loaded model
    """
    
    model_name = Path(model_path).stem
    
    if model_name == 'LightGBM':
        if lgb is None:
            raise ImportError("lightgbm not installed")
        
        # create wrapper and load the model
        wrapper = LightGBMWrapper(num_features=1, num_classes=config.num_classes, config=config)
        wrapper.load(model_path)
        return wrapper
    elif model_name == 'RealMLP':
        # load state dict to infer architecture parameters
        state_dict = torch.load(model_path, map_location='cpu')
        
        if num_features is None:
            # infer num_features from first layer
            for key in state_dict.keys():
                if 'weight' in key and len(state_dict[key].shape) == 2:
                    num_features = state_dict[key].shape[1]
                    break
        
        if num_features is None:
            raise ValueError("Could not determine input dimensions for RealMLP")
        
        # infer hidden_units from first block
        hidden_units = 256  # default
        for key in state_dict.keys():
            if 'model.blocks.0.linear.weight' in key:
                hidden_units = state_dict[key].shape[0]
                break
        
        # infer num_layers by counting blocks
        num_layers = 3  # default
        max_block_idx = -1
        for key in state_dict.keys():
            if 'model.blocks.' in key and '.linear.weight' in key:
                block_idx = int(key.split('.')[2])
                max_block_idx = max(max_block_idx, block_idx)
        if max_block_idx >= 0:
            num_layers = max_block_idx + 1
        
        # create model and load state
        model = RealMLP(
            num_features=num_features,
            hidden_units=hidden_units,
            num_classes=config.num_classes,
            dropout=getattr(config, 'realmlp_dropout', 0.1),
            num_layers=num_layers
        )
        model.load_state_dict(torch.load(model_path, map_location=config.get_device()))
        model.to(config.get_device())
        model.eval()
        return model
    elif model_name == 'TabM':
        # load state dict to infer architecture parameters
        state_dict = torch.load(model_path, map_location='cpu')
        
        if num_features is None:
            # infer num_features from first layer
            for key in state_dict.keys():
                if 'weight' in key and len(state_dict[key].shape) == 2:
                    num_features = state_dict[key].shape[1]
                    break
        
        if num_features is None:
            raise ValueError("Could not determine input dimensions for TabM")
        
        # infer d_block from model architecture
        d_block = 256  # default
        for key in state_dict.keys():
            if 'model.backbone.blocks.0.0.s' in key:
                d_block = state_dict[key].shape[1]
                break
        
        # infer n_blocks from model architecture
        n_blocks = 3  # default
        max_block_idx = -1
        for key in state_dict.keys():
            if 'model.backbone.blocks.' in key:
                block_idx = int(key.split('.')[3])
                max_block_idx = max(max_block_idx, block_idx)
        if max_block_idx >= 0:
            n_blocks = max_block_idx + 1
        
        # infer k from model architecture
        k = 8  # default
        for key in state_dict.keys():
            if 'model.backbone.blocks.0.0.r' in key:
                k = state_dict[key].shape[0]
                break
        
        # create model and load state
        model = TabMWrapper(
            num_features=num_features,
            num_classes=config.num_classes,
            n_blocks=n_blocks,
            d_block=d_block,
            k=k,
            dropout=getattr(config, 'tabm_dropout', 0.0)
        )
        model.load_state_dict(torch.load(model_path, map_location=config.get_device()))
        model.to(config.get_device())
        model.eval()
        return model
    else:
        raise ValueError(f"Unknown tabular model: {model_name}")