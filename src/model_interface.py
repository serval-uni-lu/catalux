import pickle
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import torch
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .config import Config
from .gnn_models import get_gnn_constructors, load_pretrained, predict_for_ids
from .gnn_train import GNNTrainer
from .tab_models import get_tabular_model_constructors, load_pretrained_tabular
from .tab_train import TabularTrainer


class BaseModel(ABC):
    """Abstract base class for all fraud detection models."""
    
    @abstractmethod
    def train(self, data, labels, **kwargs) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, data, **kwargs) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def evaluate(self, data, labels, **kwargs) -> Dict[str, float]:
        """Evaluate model performance."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model from disk."""
        pass


class GNNModel(BaseModel):
    """Wrapper for Graph Neural Network models."""
    
    def __init__(self, model_name: str, config: Config):
        """Initialize GNN model.
        
        Args:
            model_name: Name of the GNN model (GCN, GAT, etc.)
            config: Configuration object
        """
        self.model_name = model_name
        self.config = config
        self.device = config.get_device()
        self.model = None
        self.trainer = None
        
    def train(
        self,
        graph_data,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """Train the GNN model.
        
        Args:
            graph_data: PyTorch Geometric Data object
            train_mask: Training node mask
            val_mask: Validation node mask
            
        Returns:
            Training history and metrics
        """
        if self.trainer is None:
            self.trainer = GNNTrainer(self.config)
        
        # get model constructor
        edge_dim = graph_data.edge_attr.shape[1] if graph_data.edge_attr is not None else 0
        constructors = get_gnn_constructors(graph_data.x.shape[1], edge_dim, self.config)
        
        if self.model_name not in constructors:
            raise ValueError(f"Unknown GNN model: {self.model_name}")
        
        # train model
        self.model = constructors[self.model_name]()
        self.model, history = self.trainer.train_model(
            self.model, graph_data, train_mask, val_mask, self.model_name
        )
        
        return history
    
    def predict(self, graph_data, node_ids: Optional[List[int]] = None, **kwargs) -> np.ndarray:
        """Make predictions for nodes.
        
        Args:
            graph_data: PyTorch Geometric Data object
            node_ids: Specific node IDs to predict (None for all)
            
        Returns:
            Fraud probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if node_ids is None:
            node_ids = list(range(graph_data.num_nodes))
        
        return predict_for_ids(self.model, graph_data, node_ids)
    
    def evaluate(self, graph_data, test_mask: torch.Tensor, **kwargs) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            graph_data: PyTorch Geometric Data object
            test_mask: Test node mask
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if self.trainer is None:
            self.trainer = GNNTrainer(self.config)
        
        return self.trainer.evaluate(self.model, graph_data, test_mask)
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved {self.model_name} to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        logger.warning("Using load() without proper dimensions. Use load_model_with_data() instead.")
        try:
            self.model = load_pretrained(path, num_features=1, edge_dim=1, config=self.config)
            logger.info(f"Loaded {self.model_name} from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


class TabularModel(BaseModel):
    """Wrapper for Tabular models."""
    
    def __init__(self, model_name: str, config: Config):
        """Initialize tabular model.
        
        Args:
            model_name: Name of the tabular model (RealMLP, TabM, etc.)
            config: Configuration object
        """
        self.model_name = model_name
        self.config = config
        self.device = config.get_device()
        self.model = None
        self.trainer = None
        
    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_size: float = 0.1,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Train the tabular model.
        
        Args:
            features: Node features
            labels: Node labels
            val_size: Validation split size
            test_size: Test split size
            
        Returns:
            Training history and metrics
        """
        if self.trainer is None:
            self.trainer = TabularTrainer(self.config)
        
        # get model constructor
        constructors = get_tabular_model_constructors(features.shape[1], self.config)
        
        if self.model_name not in constructors:
            raise ValueError(f"Unknown tabular model: {self.model_name}")
        
        # train model using the trainer
        node_features_df = pd.DataFrame(features)
        node_features_df['scam'] = labels
        
        results = self.trainer.train_models(
            [self.model_name], node_features_df, labels,
            val_size=val_size, test_size=test_size,
            save_models=False
        )
        
        if self.model_name in results and 'error' not in results[self.model_name]:
            self.model = results[self.model_name]['model']
            return results[self.model_name].get('history', {})
        else:
            raise RuntimeError(f"Failed to train {self.model_name}")
    
    def predict(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions.
        
        Args:
            features: Node features
            
        Returns:
            Fraud probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # handle different model types
        if hasattr(self.model, 'predict_proba'):
            # sklearn
            probs = self.model.predict_proba(features)
            return probs[:, 1] if probs.ndim > 1 else probs
        elif hasattr(self.model, 'forward'):
            # pyTorch
            self.model.eval()
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32, device=self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                return probs.cpu().numpy()
        else:
            raise NotImplementedError(f"Prediction not implemented for {type(self.model)}")
    
    def evaluate(self, features: np.ndarray, labels: np.ndarray, **kwargs) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            features: Node features
            labels: Node labels
            
        Returns:
            Evaluation metrics
        """
        predictions = self.predict(features)
        
        pred_labels = (predictions > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(labels, pred_labels),
            'f1': f1_score(labels, pred_labels),
            'precision': precision_score(labels, pred_labels, zero_division=0),
            'recall': recall_score(labels, pred_labels, zero_division=0),
            'auc': roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
        }
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.model, 'save_model'):
            # sklearn
            self.model.save_model(path)
        elif hasattr(self.model, 'state_dict'):
            # pyTorch
            torch.save(self.model.state_dict(), path)
        else:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
        
        logger.info(f"Saved {self.model_name} to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = load_pretrained_tabular(path, self.config)
        logger.info(f"Loaded {self.model_name} from {path}")


class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_model(
        model_name: str,
        config: Config,
        model_type: Optional[str] = None
    ) -> BaseModel:
        """Create a model instance.
        
        Args:
            model_name: Name of the model
            config: Configuration object
            model_type: Type of model ('gnn' or 'tabular', auto-detected if None)
            
        Returns:
            Model instance
        """
        gnn_models = ['GCN', 'GAT', 'GATv2', 'SAGE', 'Chebyshev']
        
        if model_type == 'gnn' or model_name in gnn_models:
            return GNNModel(model_name, config)
        elif model_type == 'tabular' or model_name not in gnn_models:
            return TabularModel(model_name, config)
        else:
            raise ValueError(f"Cannot determine model type for {model_name}")
    
    @staticmethod
    def load_model(model_path: str, config: Config) -> BaseModel:
        """Load a model from disk.
        
        Args:
            model_path: Path to saved model
            config: Configuration object
            
        Returns:
            Loaded model instance
        """
        # infer model type from filename
        model_name = Path(model_path).stem
        model = ModelFactory.create_model(model_name, config)
        model.load(model_path)
        return model
    
    @staticmethod
    def load_model_with_data(
        model_path: str,
        config: Config,
        node_features,
        edge_features
    ) -> BaseModel:
        """Load a model from disk with proper data dimensions.
        
        Args:
            model_path: Path to saved model
            config: Configuration object
            node_features: Node features DataFrame
            edge_features: Edge features DataFrame
            
        Returns:
            Loaded model instance
        """
        model_name = Path(model_path).stem
        
        # check if it's a GNN model
        gnn_models = ['GCN', 'GAT', 'GATv2', 'SAGE', 'Chebyshev']
        feature_names = [col for col in node_features.columns 
                        if col not in ['node_id', 'scam', 'address', 'scam_category']]
        num_features = len(feature_names)
        
        if model_name in gnn_models:
            # load pretrained GNN
            edge_dim = edge_features.shape[1] if edge_features is not None else 0
            model = GNNModel(model_name, config)
            model.model = load_pretrained(model_path, num_features, edge_dim, config)
            return model
        else:
            # load pretrained tabular model
            model = TabularModel(model_name, config)
            model.model = load_pretrained_tabular(model_path, config, num_features)
            return model