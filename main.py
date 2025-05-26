import os
import torch
import argparse
import pandas as pd

from src.utils import load_config, seed_everything
from src.loader import load_dataset, data_to_pyg
from src.models import *
from src.train import train, test

from sklearn.model_selection import StratifiedKFold

from loguru import logger

def main():
    
    # parse
    parser = argparse.ArgumentParser(description="Train GNN models on graph datasets.")
    parser.add_argument('-d', '--dataset', choices=['ammari', 'etfd'], default='ammari',
                        help='Dataset to use: "ammari" or "etfd" (default: ammari)')
    args = parser.parse_args()

    # config
    config = load_config()
    hidden_units = config['hidden_units']
    num_classes = config['num_classes']
    num_folds = config.get('num_folds', 5)
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    seed_everything(config['seed'])
    
    print("Configuration:")
    max_key_len = max(len(str(key)) for key in config.keys())
    for key, value in config.items():
        print(f"{key:<{max_key_len}} : {value}")
    
    # load data
    features, edges = load_dataset(args.dataset)
    if num_folds == 1:
        data = data_to_pyg(features, edges)
    else:
        data = data_to_pyg(features, edges, val_ratio=1.0/num_folds, test_ratio=0.0)
    data.to(device)
    num_features = data.num_features
    logger.info('Graph data loaded successfully')

    # training
    model_constructors = get_model_constructors(num_features, hidden_units, num_classes, config.get('chebyshev_k'))

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{args.dataset}_cv_results.csv")

    all_results_accumulator = []

    for model_name, model_constructor in model_constructors.items():
        logger.info(f"Processing model: {model_name}")

        # single fold
        if num_folds == 1:
            logger.info(f"Performing a single run for {model_name} (num_folds=1)")
            model_instance = model_constructor().to(device)
            trained_model = train(config, model_instance, data) 
            precision, recall, f1 = test(trained_model, data)
            
            all_results_accumulator.append({
                "model": model_name, 
                "fold": "N/A",
                "precision": precision, 
                "recall": recall, 
                "f1": f1
            })
            logger.info(f"Finished single run for {model_name}. Test Results: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            
            model_instance.cpu()
            torch.cuda.empty_cache()
            continue

        # cross-validation
        logger.info(f"Starting {num_folds}-fold cross-validation for {model_name}")
        
        original_train_indices = data.train_mask.nonzero(as_tuple=False).view(-1)
        original_val_indices = data.val_mask.nonzero(as_tuple=False).view(-1)
        
        cv_node_indices = torch.cat([original_train_indices, original_val_indices]).unique()
        cv_node_indices_np = cv_node_indices.cpu().numpy()

        if len(cv_node_indices_np) < num_folds:
            logger.warning(
                f"Not enough nodes ({len(cv_node_indices_np)}) for {num_folds}-fold CV for model {model_name}. "
                f"Skipping CV for this model. Minimum {num_folds} samples are required."
            )
            all_results_accumulator.append({
                "model": model_name, "fold": f"skipped_low_samples ({len(cv_node_indices_np)}<{num_folds})", 
                "precision": float('nan'), "recall": float('nan'), "f1": float('nan')
            })
            continue

        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config['seed'])
        y_cv = data.y[cv_node_indices].cpu().numpy()
        model_fold_metrics = []

        for fold_idx, (train_fold_local_indices, val_fold_local_indices) in enumerate(kf.split(cv_node_indices_np, y_cv)):
            logger.info(f"Training {model_name} - Fold {fold_idx + 1}/{num_folds}")
            
            model_instance = model_constructor().to(device)
            fold_data = data.clone()
            fold_data.to(device)
            
            train_global_indices = cv_node_indices[train_fold_local_indices]
            val_global_indices = cv_node_indices[val_fold_local_indices]

            fold_data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            fold_data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            fold_data.train_mask[train_global_indices] = True
            fold_data.val_mask[val_global_indices] = True
            
            assert not (fold_data.train_mask & fold_data.val_mask).any(), "Train and Val masks overlap!"
            assert not (fold_data.train_mask & fold_data.test_mask).any(), "Train and Test masks overlap!"
            assert not (fold_data.val_mask & fold_data.test_mask).any(), "Val and Test masks overlap!"
            
            trained_model_fold = train(config, model_instance, fold_data)
            precision, recall, f1 = test(trained_model_fold, fold_data, mask=fold_data.val_mask)
            
            model_fold_metrics.append({'precision': precision, 'recall': recall, 'f1': f1})
            all_results_accumulator.append({
                "model": model_name, 
                "fold": fold_idx + 1,
                "precision": precision, 
                "recall": recall, 
                "f1": f1
            })
            logger.info(f"Fold {fold_idx + 1}/{num_folds} for {model_name}: Test Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            
            model_instance.cpu()
            torch.cuda.empty_cache()

        if model_fold_metrics:
            avg_precision = sum(m['precision'] for m in model_fold_metrics) / num_folds
            avg_recall = sum(m['recall'] for m in model_fold_metrics) / num_folds
            avg_f1 = sum(m['f1'] for m in model_fold_metrics) / num_folds
            
            logger.info(f"Avg Test Results for {model_name} over {num_folds} folds: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}")
            all_results_accumulator.append({
                "model": model_name,
                "fold": "average",
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            })

    results_df = pd.DataFrame(all_results_accumulator)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Cross-validation results saved to {results_path}")

if __name__ == "__main__":
    main()