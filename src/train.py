import time
import torch
import numpy as np
import torch.nn.functional as F

from loguru import logger
from torch.nn import Module
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .config import Config
from .models import get_model_constructors, GATConvolution, GATv2Convolution

class GNNTrainer:
    """Main trainer class for GNN models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = self._setup_device()
        self._set_seed()
        
    def _setup_device(self) -> torch.device:
        """Setup computing device."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)
    
    def _set_seed(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
    
    def train_epoch(self, model: Module, data: Data, optimizer: torch.optim.Optimizer,
                   train_mask: torch.Tensor) -> float:
        """Train for one epoch."""
        model.train()
        optimizer.zero_grad()
        
        if isinstance(model, (GATConvolution, GATv2Convolution)):
            out = model(data.x, data.edge_index, data.edge_attr)
        else:
            out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, model: Module, data: Data, mask: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance."""
        model.eval()
        
        with torch.no_grad():
            if isinstance(model, (GATConvolution, GATv2Convolution)):
                out = model(data.x, data.edge_index, data.edge_attr)
            else:
                out = model(data.x, data.edge_index)
            pred = out[mask].argmax(dim=1)
            true = data.y[mask]
            
            pred_np = pred.cpu().numpy()
            true_np = true.cpu().numpy()
            
            probs = F.softmax(out[mask], dim=1)
            probs_np = probs.cpu().numpy()
            
            metrics = {
                'accuracy': accuracy_score(true_np, pred_np),
                'f1': f1_score(true_np, pred_np, average='macro'),
                'precision': precision_score(true_np, pred_np, average='macro', zero_division=0),
                'recall': recall_score(true_np, pred_np, average='macro', zero_division=0),
                'loss': F.nll_loss(out[mask], true).item()
            }
            if self.config.num_classes == 2:
                metrics['auc'] = roc_auc_score(true_np, probs_np[:, 1])
            
        return metrics
    
    def train_model(self, model: Module, data: Data, train_mask: torch.Tensor,
                   val_mask: torch.Tensor) -> Tuple[Module, Dict[str, List[float]]]:
        """Train a single model with early stopping."""
        model = model.to(self.device)
        data = data.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr,
                                   weight_decay=self.config.weight_decay)
        
        best_val_f1 = 0
        best_model_state = None
        patience_counter = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': []
        }
        
        logger.info(f"Training on {self.device}")
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(model, data, optimizer, train_mask)
            
            val_metrics = self.evaluate(model, data, val_mask)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1'])
            
            # early stopping
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                          f"Val F1: {val_metrics['f1']:.4f}")
        
        model.load_state_dict(best_model_state)
        return model, history
    
    def cross_validate(self, model_name: str, data: Data) -> Dict[str, Any]:
        """Perform k-fold cross validation."""
        logger.info(f"Starting cross-validation for {model_name}")
        
        constructors = get_model_constructors(data.x.shape[1], self.config)
        if model_name not in constructors:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_constructor = constructors[model_name]
        
        y_np = data.y.cpu().numpy()
        skf = StratifiedKFold(n_splits=self.config.num_folds, shuffle=True, 
                            random_state=self.config.seed)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(y_np)), y_np)):
            logger.info(f"Fold {fold + 1}/{self.config.num_folds}")
            
            train_mask = torch.zeros(len(y_np), dtype=torch.bool)
            val_mask = torch.zeros(len(y_np), dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            
            model = model_constructor()
            
            # training
            start_time = time.time()
            trained_model, history = self.train_model(model, data, train_mask, val_mask)
            training_time = time.time() - start_time

            # evaluation
            val_metrics = self.evaluate(trained_model, data, val_mask)
            val_metrics['training_time'] = training_time
            val_metrics['fold'] = fold
            
            fold_results.append(val_metrics)
            
            logger.info(f"Fold {fold + 1} - Val Acc: {val_metrics['accuracy']:.4f}, "
                      f"Val F1: {val_metrics['f1']:.4f}")
        
        metrics_keys = ['accuracy', 'f1', 'precision', 'recall', 'loss', 'training_time']
        if self.config.num_classes == 2:
            metrics_keys.append('auc')
        
        aggregated_results = {}
        for key in metrics_keys:
            values = [result[key] for result in fold_results]
            aggregated_results[f'{key}_mean'] = np.mean(values)
            aggregated_results[f'{key}_std'] = np.std(values)
        
        aggregated_results['fold_results'] = fold_results
        
        return aggregated_results
    
    def evaluate_all_models(self, data: Data) -> Dict[str, Dict[str, Any]]:
        """Evaluate all available models using cross-validation."""
        constructors = get_model_constructors(data.x.shape[1], self.config)
        results = {}
        
        for model_name in constructors.keys():
            logger.info(f"Evaluating {model_name}")
            try:
                results[model_name] = self.cross_validate(model_name, data)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results