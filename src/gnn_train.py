import os
import sys
import argparse
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .config import Config
from .utils import seed_everything
from .gnn_models import get_gnn_constructors, GAT, GATv2
from .dataloader import load_txs, to_account_features, to_pyg_data


class GNNTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.get_device()

    def train_epoch(self, model: Module, data: Data, optimizer: torch.optim.Optimizer,
                   train_mask: torch.Tensor) -> float:
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr) if isinstance(model, (GAT, GATv2)) else model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def train_model(self, model: Module, data: Data, train_mask: torch.Tensor,
                   val_mask: torch.Tensor, model_name: str = None) -> Tuple[Module, Dict[str, List[float]]]:
        model, data = model.to(self.device), data.to(self.device)
        
        # get model specific learning rate
        lr = self.config.lr
        if model_name:
            lr = getattr(self.config, f'{model_name.lower()}_lr', self.config.lr)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.config.weight_decay)
        best_val_f1, best_model_state, patience_counter = 0, model.state_dict().copy(), 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        logger.info(f"Training on {self.device}")
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(model, data, optimizer, train_mask)
            val_metrics = self.evaluate(model, data, val_mask)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1'])
            
            if val_metrics['f1'] > best_val_f1:
                best_val_f1, best_model_state, patience_counter = val_metrics['f1'], model.state_dict().copy(), 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        model.load_state_dict(best_model_state)
        return model, history
    
    def evaluate(self, model: Module, data: Data, mask: torch.Tensor) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr) if isinstance(model, (GAT, GATv2)) else model(data.x, data.edge_index)
            loss = F.nll_loss(out[mask], data.y[mask])
            pred = out[mask].max(1)[1]
            y_true, y_pred = data.y[mask].cpu().numpy(), pred.cpu().numpy()
            y_prob = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
            }
    
    def create_masks(self, data: Data, val_size: float = 0.1, test_size: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = data.x.shape[0] if data.x is not None else data.num_nodes
        indices = np.arange(num_nodes)
        train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=data.y.cpu())
        train_idx, val_idx = train_test_split(train_idx, test_size=val_size/(1-test_size), stratify=data.y[train_idx].cpu())
        
        masks = [torch.zeros(num_nodes, dtype=torch.bool) for _ in range(3)]
        masks[0][train_idx], masks[1][val_idx], masks[2][test_idx] = True, True, True
        return tuple(m.to(self.device) for m in masks)
    
    def save_model(self, model: Module, model_name: str, models_dir: str = "models") -> str:
        Path(models_dir).mkdir(exist_ok=True)
        model_path = os.path.join(models_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved {model_name} to {model_path}")
        return model_path

    def train_models(self, model_names: List[str], data: Data, val_size: float = 0.1, test_size: float = 0.2, save_models: bool = True) -> Dict[str, Dict[str, Any]]:
        edge_dim = data.edge_attr.shape[1] if data.edge_attr is not None else 0
        constructors = get_gnn_constructors(data.x.shape[1], edge_dim, self.config)
        train_mask, val_mask, test_mask = self.create_masks(data, val_size, test_size)
        results = {}
        
        for model_name in model_names:
            if model_name not in constructors:
                logger.error(f"Model {model_name} not available. Available: {list(constructors.keys())}")
                continue
                
            logger.info(f"Training {model_name}")
            try:
                model, history = self.train_model(constructors[model_name](), data, train_mask, val_mask, model_name)
                test_metrics = self.evaluate(model, data, test_mask)
                results[model_name] = {'model': model, 'history': history, 'test_metrics': test_metrics}
                
                if save_models:
                    results[model_name]['model_path'] = self.save_model(model, model_name)
                    
                logger.info(f"{model_name} - Test F1: {test_metrics['f1']:.4f}, Test AUC: {test_metrics['auc']:.4f}")

                # clear GPU memory after each model
                if torch.cuda.is_available():
                    del model
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}

                # clear GPU memory even on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Train GNN models for fraud detection')
    parser.add_argument('--models', nargs='+', help='Model names to train (e.g., GCN GAT) or "all" for all models')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--dataset', default='etfd', help='Dataset name')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation split size (default: 0.1)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split size (default: 0.2)')
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    seed_everything(config.seed)
    
    logger.info(f"Loading dataset: {args.dataset}")
    txs = load_txs(args.dataset)
    node_features, edges, edge_features = to_account_features(txs=txs, use_address=True, scam_features=True, edge_features=True)
    
    node_labels = node_features['scam'].values
    feature_names = [col for col in node_features.columns if col not in ['node_id', 'scam', 'address', 'scam_category']]
    
    node_scaler, edge_scaler = StandardScaler(), StandardScaler()
    scaled_node_features = node_scaler.fit_transform(node_features[feature_names].values)
    scaled_edge_features = edge_scaler.fit_transform(edge_features.values) if edge_features is not None else None
    
    graph = to_pyg_data(scaled_node_features, node_labels, edges.values, scaled_edge_features, config.get_device())
    logger.info(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    all_models = ['GCN', 'GAT', 'GATv2', 'SAGE', 'Chebyshev']
    models_to_train = all_models if args.models == ['all'] else args.models
    
    if invalid := [m for m in models_to_train if m not in all_models]:
        logger.error(f"Invalid model names: {invalid}. Available: {all_models}")
        sys.exit(1)
    
    logger.info(f"Training models: {models_to_train}")
    trainer = GNNTrainer(config)
    results = trainer.train_models(models_to_train, graph, val_size=args.val_size, test_size=args.test_size)
    
    logger.info("Training completed!")
    for model_name, result in results.items():
        if 'error' in result:
            logger.error(f"{model_name}: {result['error']}")
        else:
            metrics = result['test_metrics']
            logger.info(f"{model_name}: F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")

if __name__ == "__main__":
    main()