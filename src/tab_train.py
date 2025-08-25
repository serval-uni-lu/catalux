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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .config import Config
from .utils import seed_everything
from .dataloader import load_txs, to_account_features
from .tab_models import get_tabular_model_constructors, LightGBMWrapper


class TabularTrainer:
    """Trainer for tabular models (RealMLP, TabM, LightGBM)."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.get_device()
    
    def prepare_tabular_data(self, node_features, node_labels, 
                            val_size: float = 0.1, test_size: float = 0.2) -> Tuple:
        """Prepare data for tabular models."""
        
        feature_cols = [col for col in node_features.columns 
                       if col not in ['node_id', 'scam', 'address', 'scam_category']]
        
        X = node_features[feature_cols].values
        y = node_labels
        
        # split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.config.seed
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            stratify=y_temp, random_state=self.config.seed
        )
        
        # scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler
    
    def train_pytorch_model(self, model: Module, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray, 
                           model_name: str = None) -> Tuple[Module, Dict[str, List[float]]]:
        """Train PyTorch-based tabular models (RealMLP, TabM)."""
        
        # convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        model = model.to(self.device)
        
        # get model-specific learning rate and batch size
        if model_name == 'TabM':
            lr = getattr(self.config, 'tabm_lr', 3e-4)
            batch_size = getattr(self.config, 'tabm_batch_size', 256)
            weight_decay = getattr(self.config, 'tabm_weight_decay', 0.0)
        elif model_name == 'RealMLP':
            lr = getattr(self.config, 'realmlp_lr', 1e-3)
            batch_size = getattr(self.config, 'realmlp_batch_size', 256)
            weight_decay = getattr(self.config, 'realmlp_weight_decay', 1e-5)
        else:
            lr = self.config.lr
            batch_size = 256
            weight_decay = self.config.weight_decay
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs, eta_min=lr * 0.01
        )
        
        best_val_f1 = 0
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        logger.info(f"Training {model_name} on {self.device} with lr={lr}, batch_size={batch_size}")
        
        is_tabm = model_name == 'TabM'
        n_batches = (len(X_train_t) + batch_size - 1) // batch_size
        
        for epoch in range(self.config.epochs):
            # training
            model.train()
            train_losses = []
            
            # shuffle data
            perm = torch.randperm(len(X_train_t))
            X_train_shuffled = X_train_t[perm]
            y_train_shuffled = y_train_t[perm]
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train_t))
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                optimizer.zero_grad()
                
                if is_tabm:
                    out = model(X_batch, return_ensemble=True)
                    # out shape: (batch_size, k, num_classes)
                    # calculate loss for each ensemble member and average
                    losses = []
                    
                    class_counts = torch.bincount(y_train_t)
                    class_weights = 1.0 / class_counts.float()
                    class_weights = class_weights / class_weights.sum() * len(class_weights)
                    class_weights = class_weights.to(self.device)
                    
                    for k in range(out.shape[1]):
                        ensemble_out = out[:, k, :]
                        loss_k = F.cross_entropy(ensemble_out, y_batch, weight=class_weights)
                        losses.append(loss_k)
                    loss = torch.mean(torch.stack(losses))
                else:
                    out = model(X_batch)
                    loss = F.nll_loss(out, y_batch)
                
                loss.backward()
                
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            scheduler.step()
            
            # validation
            val_metrics = self.evaluate_pytorch_model(model, X_val_t, y_val_t, model_name)
            
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
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        # load best model
        model.load_state_dict(best_model_state)
        return model, history
    
    def evaluate_pytorch_model(self, model: Module, X: torch.Tensor, y: torch.Tensor, 
                              model_name: str = None) -> Dict[str, float]:
        """Evaluate PyTorch model."""
        model.eval()
        with torch.no_grad():
            if model_name == 'TabM':
                out = model(X, return_ensemble=False)
            else:
                out = model(X)
            
            probs = torch.exp(out)
            pred = probs.max(1)[1]
            y_true = y.cpu().numpy()
            y_pred = pred.cpu().numpy()
            y_prob = probs[:, 1].cpu().numpy()
            
            if model_name == 'TabM':
                loss = F.cross_entropy(out, y).item()
            else:
                loss = F.nll_loss(out, y).item()
            
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
                'loss': loss
            }
    
    def train_lightgbm_model(self, model: LightGBMWrapper, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Tuple[LightGBMWrapper, Dict[str, List[float]]]:
        """Train LightGBM model."""
        
        logger.info("Training LightGBM")
        
        # train model
        model.train(
            X_train, y_train, X_val, y_val,
            num_boost_round=1000,
            early_stopping_rounds=self.config.early_stopping_patience
        )
        
        # get training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': []
        }
        
        # evaluate on validation set
        y_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, y_pred)
        history['val_f1'].append(val_f1)
        
        logger.info(f"LightGBM - Val F1: {val_f1:.4f}")
        
        return model, history
    
    def evaluate_lightgbm_model(self, model: LightGBMWrapper, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate LightGBM model."""
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0
        }
    
    def save_model(self, model, model_name: str, models_dir: str = "models") -> str:
        """Save model to file."""
        Path(models_dir).mkdir(exist_ok=True)
        
        if isinstance(model, LightGBMWrapper):
            # save lightGBM model
            model_path = os.path.join(models_dir, f"{model_name}.lgb")
            model.save(model_path)
        else:
            # save pyTorch model
            model_path = os.path.join(models_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
        
        logger.info(f"Saved {model_name} to {model_path}")
        return model_path
    
    def train_models(self, model_names: List[str], node_features, node_labels,
                    val_size: float = 0.1, test_size: float = 0.2, 
                    save_models: bool = True) -> Dict[str, Dict[str, Any]]:
        """Train multiple tabular models."""
        
        # prepare data
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = self.prepare_tabular_data(
            node_features, node_labels, val_size, test_size
        )
        
        logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # get model constructors
        constructors = get_tabular_model_constructors(X_train.shape[1], self.config)
        results = {}
        
        for model_name in model_names:
            if model_name not in constructors:
                logger.error(f"Model {model_name} not available. Available: {list(constructors.keys())}")
                continue
            
            logger.info(f"Training {model_name}")
            try:
                model = constructors[model_name]()
                
                if isinstance(model, LightGBMWrapper):
                    # train lightGBM
                    model, history = self.train_lightgbm_model(
                        model, X_train, y_train, X_val, y_val
                    )
                    test_metrics = self.evaluate_lightgbm_model(model, X_test, y_test)
                else:
                    # train pyTorch model
                    model, history = self.train_pytorch_model(
                        model, X_train, y_train, X_val, y_val, model_name
                    )
                    X_test_t = torch.FloatTensor(X_test).to(self.device)
                    y_test_t = torch.LongTensor(y_test).to(self.device)
                    test_metrics = self.evaluate_pytorch_model(model, X_test_t, y_test_t, model_name)
                
                results[model_name] = {
                    'model': model,
                    'history': history,
                    'test_metrics': test_metrics,
                    'scaler': scaler
                }
                
                if save_models:
                    results[model_name]['model_path'] = self.save_model(model, model_name)
                
                logger.info(f"{model_name} - Test F1: {test_metrics['f1']:.4f}, Test AUC: {test_metrics['auc']:.4f}")
                
                # clear GPU memory after each model
                if torch.cuda.is_available() and not isinstance(model, LightGBMWrapper):
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
    parser = argparse.ArgumentParser(description='Train tabular models for fraud detection')
    parser.add_argument('--models', nargs='+', help='Model names to train (e.g., RealMLP TabM LightGBM) or "all" for all models')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--dataset', default='etfd', help='Dataset name')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation split size (default: 0.1)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split size (default: 0.2)')
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    seed_everything(config.seed)
    
    logger.info(f"Loading dataset: {args.dataset}")
    txs = load_txs(args.dataset)
    node_features, edges, edge_features = to_account_features(
        txs=txs, use_address=True, scam_features=True, edge_features=True
    )
    feature_names = [col for col in node_features.columns 
                    if col not in ['node_id', 'scam', 'address', 'scam_category']]
    num_features = len(feature_names)
    node_labels = node_features['scam'].values
    
    # get available models
    dummy_constructors = get_tabular_model_constructors(num_features, config)
    all_models = list(dummy_constructors.keys())
    
    if args.models == ['all']:
        models_to_train = all_models
    else:
        models_to_train = args.models if args.models else all_models
    
    # validate model names
    if invalid := [m for m in models_to_train if m not in all_models]:
        logger.error(f"Invalid model names: {invalid}. Available: {all_models}")
        sys.exit(1)
    
    logger.info(f"Training models: {models_to_train}")
    trainer = TabularTrainer(config)
    results = trainer.train_models(
        models_to_train, node_features, node_labels,
        val_size=args.val_size, test_size=args.test_size
    )
    
    logger.info("Training completed!")
    for model_name, result in results.items():
        if 'error' in result:
            logger.error(f"{model_name}: {result['error']}")
        else:
            metrics = result['test_metrics']
            logger.info(f"{model_name}: F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")


if __name__ == "__main__":
    main()