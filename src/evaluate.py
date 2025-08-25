import torch
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, auc
)

from .config import Config
from .model_interface import ModelFactory
from .dataloader import load_txs, to_account_features, to_pyg_data


def evaluate_model_by_name(
    model_name: str,
    dataset: str,
    config: Config,
    test_size: float = 0.2,
    models_dir: str = "models"
) -> Dict[str, float]:
    """Evaluate a trained model by name.
    
    Args:
        model_name: Name of the model
        dataset: Dataset name
        config: Configuration object
        test_size: Test split size
        models_dir: Directory containing model files
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # determine model path based on model type
    if model_name == 'LightGBM':
        model_path = f"{models_dir}/{model_name}.lgb"
    else:
        model_path = f"{models_dir}/{model_name}.pth"
    
    return evaluate_model(model_path, dataset, config, test_size)


def evaluate_model(
    model_path: str,
    dataset: str,
    config: Config,
    test_size: float = 0.2
) -> Dict[str, float]:
    """Evaluate a trained model on test data.
    
    Args:
        model_path: Path to saved model
        dataset: Dataset name
        config: Configuration object
        test_size: Test split size
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # get model name from path
    model_name = model_path.split('/')[-1].split('.')[0]
    
    # load and preprocess data first to get correct dimensions
    txs = load_txs(dataset)
    node_features, edges, edge_features = to_account_features(
        txs=txs, use_address=True, scam_features=True, edge_features=True
    )
    
    # load model with correct dimensions
    model = ModelFactory.load_model_with_data(model_path, config, node_features, edge_features)
    
    node_labels = node_features['scam'].values
    feature_names = [col for col in node_features.columns 
                    if col not in ['node_id', 'scam', 'address', 'scam_category']]
    
    X = node_features[feature_names].values
    y = node_labels
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1/(1-test_size), 
        stratify=y_temp, random_state=config.seed
    )
    
    # get indices for the test set
    test_idx = np.arange(len(y_test))
    
    # check if model is GNN or tabular
    is_gnn = model_name in ['GCN', 'GAT', 'GATv2', 'SAGE', 'Chebyshev']
    
    if is_gnn:
        # prepare graph data
        node_scaler = StandardScaler()
        edge_scaler = StandardScaler()
        scaled_node_features = node_scaler.fit_transform(node_features[feature_names].values)
        scaled_edge_features = edge_scaler.fit_transform(edge_features.values) if edge_features is not None else None
        
        graph = to_pyg_data(
            scaled_node_features, node_labels, edges.values,
            scaled_edge_features, config.get_device()
        )
        
        # create test mask
        test_mask = torch.zeros(len(node_labels), dtype=torch.bool)
        test_mask[test_idx] = True
        test_mask = test_mask.to(config.get_device())
        
        # evaluate
        metrics = model.evaluate(graph, test_mask)
    else:
        # tabular model, use exact same preprocessing as training
        test_features = X_test
        test_labels = y_test
        
        # apply same scaling as training
        scaler = StandardScaler()
        scaler.fit(X_train)
        test_features = scaler.transform(test_features)
        
        # use consistent evaluation approach based on model type
        if hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
            # lightGBM or sklearn-style models
            pred_proba = model.model.predict_proba(test_features)[:, 1]
            pred_labels = (pred_proba > 0.5).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(test_labels, pred_labels),
                'f1': f1_score(test_labels, pred_labels),
                'precision': precision_score(test_labels, pred_labels, zero_division=0),
                'recall': recall_score(test_labels, pred_labels, zero_division=0),
                'auc': roc_auc_score(test_labels, pred_proba) if len(np.unique(test_labels)) > 1 else 0.0
            }
        elif hasattr(model, 'model') and hasattr(model.model, 'forward'):
            # pyTorch models (RealMLP, TabM)
            model.model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(test_features, dtype=torch.float32, device=config.get_device())
                
                if hasattr(model.model, 'k'):
                    logits = model.model(x_tensor, return_ensemble=False)
                else:
                    logits = model.model(x_tensor)
                
                probs = torch.exp(logits)
                pred_proba = probs[:, 1].cpu().numpy()
                pred_labels = (pred_proba > 0.5).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(test_labels, pred_labels),
                'f1': f1_score(test_labels, pred_labels),
                'precision': precision_score(test_labels, pred_labels, zero_division=0),
                'recall': recall_score(test_labels, pred_labels, zero_division=0),
                'auc': roc_auc_score(test_labels, pred_proba) if len(np.unique(test_labels)) > 1 else 0.0
            }
        else:
            metrics = model.evaluate(test_features, test_labels)
    
    return metrics


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray
) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    if len(np.unique(y_true)) > 1:
        metrics['auc'] = roc_auc_score(y_true, y_probs)
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        metrics['auc_pr'] = auc(recall, precision)
    else:
        metrics['auc'] = metrics['auc_pr'] = 0.0
    
    # confusion matrix metrics
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0
        })
    
    return metrics


def cross_validate_model(
    model_path: str,
    dataset: str,
    config: Config,
    n_folds: int = 5
) -> Dict[str, List[float]]:
    """Perform cross-validation on a model.
    
    Args:
        model_path: Path to saved model
        dataset: Dataset name
        config: Configuration object
        n_folds: Number of cross-validation folds
        
    Returns:
        Dictionary containing metrics for each fold
    """
    # load model type
    model_name = model_path.split('/')[-1].split('.')[0]
    
    # load and preprocess data
    txs = load_txs(dataset)
    node_features, edges, edge_features = to_account_features(
        txs=txs, use_address=True, scam_features=True, edge_features=True
    )
    
    node_labels = node_features['scam'].values
    feature_names = [col for col in node_features.columns 
                    if col not in ['node_id', 'scam', 'address', 'scam_category']]
    
    # prepare for cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
    indices = np.arange(len(node_labels))
    
    cv_results = {
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'auc': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, node_labels)):
        logger.info(f"Evaluating fold {fold + 1}/{n_folds}")
        
        # create a new model instance for each fold
        model = ModelFactory.create_model(model_name, config)
        
        # check if model is GNN or tabular
        is_gnn = model_name in ['GCN', 'GAT', 'GATv2', 'SAGE', 'Chebyshev']
        
        if is_gnn:
            # prepare graph data
            node_scaler = StandardScaler()
            edge_scaler = StandardScaler()
            scaled_node_features = node_scaler.fit_transform(node_features[feature_names].values)
            scaled_edge_features = edge_scaler.fit_transform(edge_features.values) if edge_features is not None else None
            
            graph = to_pyg_data(
                scaled_node_features, node_labels, edges.values,
                scaled_edge_features, config.get_device()
            )
            
            # create masks
            train_mask = torch.zeros(len(node_labels), dtype=torch.bool)
            train_mask[train_idx] = True
            test_mask = torch.zeros(len(node_labels), dtype=torch.bool)
            test_mask[test_idx] = True
            
            # split train into train/val
            val_split = int(len(train_idx) * 0.9)
            val_mask = torch.zeros(len(node_labels), dtype=torch.bool)
            val_mask[train_idx[val_split:]] = True
            train_mask[train_idx[val_split:]] = False
            
            train_mask = train_mask.to(config.get_device())
            val_mask = val_mask.to(config.get_device())
            test_mask = test_mask.to(config.get_device())
            
            # train and evaluate
            model.train(graph, train_mask, val_mask)
            metrics = model.evaluate(graph, test_mask)
        else:
            # tabular model
            scaler = StandardScaler()
            all_features = scaler.fit_transform(node_features[feature_names].values)
            
            train_features = all_features[train_idx]
            train_labels = node_labels[train_idx]
            test_features = all_features[test_idx]
            test_labels = node_labels[test_idx]
            
            # train and evaluate
            model.train(train_features, train_labels)
            metrics = model.evaluate(test_features, test_labels)
        
        # store results
        for metric_name, value in metrics.items():
            if metric_name in cv_results:
                cv_results[metric_name].append(value)
    
    return cv_results


def print_evaluation_results(results: Dict[str, float]):
    """Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary of evaluation metrics
    """
    print("\nModel Evaluation Results:")
    print("-" * 30)
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{metric.capitalize()}: {value:.4f}")


def print_cv_results(cv_results: Dict[str, List[float]]):
    """Print cross-validation results.
    
    Args:
        cv_results: Dictionary of metrics for each fold
    """
    print("\nCross-Validation Results:")
    print("-" * 40)
    for metric, values in cv_results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")


def plot_confusion_matrix(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None
):
    """Plot confusion matrix.
    
    Args:
        true_labels: True labels
        predictions: Predicted labels
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def get_classification_report(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    target_names: Optional[List[str]] = None
) -> str:
    """Generate classification report.
    
    Args:
        true_labels: True labels
        predictions: Predicted labels
        target_names: Optional names for the classes
        
    Returns:
        Classification report string
    """
    if target_names is None:
        target_names = ['Normal', 'Fraud']
    
    return classification_report(
        true_labels, predictions,
        target_names=target_names
    )


def compare_models(
    model_paths: List[str],
    dataset: str,
    config: Config,
    test_size: float = 0.2
) -> Dict[str, Dict[str, float]]:
    """Compare multiple models on the same dataset.
    
    Args:
        model_paths: List of paths to saved models
        dataset: Dataset name
        config: Configuration object
        test_size: Test split size
        
    Returns:
        Dictionary mapping model names to their metrics
    """
    results = {}
    
    for model_path in model_paths:
        model_name = model_path.split('/')[-1].split('.')[0]
        logger.info(f"Evaluating {model_name}...")
        
        try:
            metrics = evaluate_model(model_path, dataset, config, test_size)
            results[model_name] = metrics
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # print comparison table
    if results:
        print("\nModel Comparison:")
        print("-" * 70)
        print(f"{'Model':<15} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'AUC':<10}")
        print("-" * 70)
        
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                print(f"{model_name:<15} "
                      f"{metrics.get('accuracy', 0):<10.4f} "
                      f"{metrics.get('f1', 0):<10.4f} "
                      f"{metrics.get('precision', 0):<10.4f} "
                      f"{metrics.get('recall', 0):<10.4f} "
                      f"{metrics.get('auc', 0):<10.4f}")
            else:
                print(f"{model_name:<15} Error: {metrics['error'][:40]}")
    
    return results


def save_evaluation_results(
    results: Dict[str, Dict[str, float]],
    output_path: str
):
    """Save evaluation results to CSV file.
    
    Args:
        results: Dictionary of model results
        output_path: Path to save CSV file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # convert to dataFrame
    df_data = {}
    for model_name, metrics in results.items():
        if 'error' not in metrics:
            df_data[model_name] = metrics
    
    if df_data:
        df = pd.DataFrame(df_data).T
        df.to_csv(output_path)
        logger.info(f"Evaluation results saved to {output_path}")
    else:
        logger.warning("No valid results to save")