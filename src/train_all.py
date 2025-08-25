import sys
import json
import argparse
from loguru import logger
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler

from .config import Config
from .utils import seed_everything
from .dataloader import load_txs, to_account_features, to_pyg_data

# GNN imports
from .gnn_train import GNNTrainer

# tabular imports
from .tab_train import TabularTrainer
from .tab_models import get_tabular_model_constructors


def train_all_models(
    models: List[str],
    dataset: str,
    model_type: str,
    config: Config,
    val_size: float = 0.1,
    test_size: float = 0.2,
    save_models: bool = True
) -> Dict[str, Any]:
    """Train specified models on the dataset.
    
    Args:
        models: List of model names to train
        dataset: Dataset name
        model_type: Type of models ('gnn', 'tabular', 'both')
        config: Configuration object
        val_size: Validation split size
        test_size: Test split size
        save_models: Whether to save trained models
        
    Returns:
        Dictionary containing training results for each model
    """
    logger.info(f"Loading dataset: {dataset}")
    txs = load_txs(dataset)
    node_features, edges, edge_features = to_account_features(
        txs=txs, use_address=True, scam_features=True, edge_features=True
    )
    
    node_labels = node_features['scam'].values
    feature_names = [col for col in node_features.columns 
                    if col not in ['node_id', 'scam', 'address', 'scam_category']]
    
    gnn_models = ['GCN', 'GAT', 'GATv2', 'SAGE', 'Chebyshev']
    dummy_constructors = get_tabular_model_constructors(len(feature_names), config)
    tabular_models = list(dummy_constructors.keys())
    
    if models == ['all']:
        if model_type == 'gnn':
            models = gnn_models
        elif model_type == 'tabular':
            models = tabular_models
        else:  # both
            models = gnn_models + tabular_models
    
    # separate GNN and tabular models
    gnn_to_train = [m for m in models if m in gnn_models]
    tabular_to_train = [m for m in models if m in tabular_models]
    
    # check for invalid models
    all_available = gnn_models + tabular_models
    if invalid := [m for m in models if m not in all_available]:
        logger.error(f"Invalid model names: {invalid}")
        logger.info(f"Available GNN models: {gnn_models}")
        logger.info(f"Available tabular models: {tabular_models}")
        raise ValueError(f"Invalid model names: {invalid}")
    
    all_results = {}
    
    # train GNN models
    if gnn_to_train and model_type in ['gnn', 'both']:
        logger.info(f"Training GNN models: {gnn_to_train}")
        
        # prepare graph data
        node_scaler = StandardScaler()
        edge_scaler = StandardScaler()
        scaled_node_features = node_scaler.fit_transform(node_features[feature_names].values)
        scaled_edge_features = edge_scaler.fit_transform(edge_features.values) if edge_features is not None else None
        
        graph = to_pyg_data(scaled_node_features, node_labels, edges.values, 
                           scaled_edge_features, config.get_device())
        logger.info(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
        
        # train GNN models
        gnn_trainer = GNNTrainer(config)
        gnn_results = gnn_trainer.train_models(
            gnn_to_train, graph, 
            val_size=val_size, test_size=test_size,
            save_models=save_models
        )
        
        # add to results
        for model_name, result in gnn_results.items():
            if 'error' not in result:
                all_results[model_name] = {
                    'type': 'GNN',
                    'test_metrics': result['test_metrics'],
                    'model_path': result.get('model_path', None)
                }
    
    # train tabular models
    if tabular_to_train and model_type in ['tabular', 'both']:
        logger.info(f"Training tabular models: {tabular_to_train}")
        
        # train tabular models
        tabular_trainer = TabularTrainer(config)
        tabular_results = tabular_trainer.train_models(
            tabular_to_train, node_features, node_labels,
            val_size=val_size, test_size=test_size,
            save_models=save_models
        )
        
        # add to results
        for model_name, result in tabular_results.items():
            if 'error' not in result:
                all_results[model_name] = {
                    'type': 'Tabular',
                    'test_metrics': result['test_metrics'],
                    'model_path': result.get('model_path', None)
                }
    
    return all_results


def main():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(description='Train GNN and/or tabular models for fraud detection')
    parser.add_argument('--models', nargs='+', 
                       help='Model names to train (e.g., GCN GAT RealMLP TabM LightGBM) or "all" for all models')
    parser.add_argument('--type', choices=['gnn', 'tabular', 'both'], default='both',
                       help='Type of models to train: gnn, tabular, or both (default: both)')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--dataset', default='etfd', help='Dataset name')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation split size (default: 0.1)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split size (default: 0.2)')
    parser.add_argument('--save-results', action='store_true', help='Save results to JSON file')
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    seed_everything(config.seed)
    
    # train models
    try:
        all_results = train_all_models(
            models=args.models if args.models else ['all'],
            dataset=args.dataset,
            model_type=args.type,
            config=config,
            val_size=args.val_size,
            test_size=args.test_size,
            save_models=True
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # print summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    if all_results:
        # sort by f1 score
        sorted_results = sorted(all_results.items(), 
                               key=lambda x: x[1]['test_metrics']['f1'], 
                               reverse=True)
        
        logger.info(f"{'Model':<15} {'Type':<10} {'F1 Score':<10} {'AUC':<10} {'Accuracy':<10}")
        logger.info("-"*60)
        
        for model_name, result in sorted_results:
            metrics = result['test_metrics']
            logger.info(
                f"{model_name:<15} {result['type']:<10} "
                f"{metrics['f1']:<10.4f} {metrics['auc']:<10.4f} "
                f"{metrics['accuracy']:<10.4f}"
            )
        
        # best model
        best_model = sorted_results[0]
        logger.info("\n" + "="*60)
        logger.info(f"BEST MODEL: {best_model[0]} ({best_model[1]['type']})")
        logger.info(f"F1 Score: {best_model[1]['test_metrics']['f1']:.4f}")
        logger.info(f"AUC: {best_model[1]['test_metrics']['auc']:.4f}")
        logger.info("="*60)
        
        # save results if requested
        if args.save_results:
            results_file = 'training_results.json'
            with open(results_file, 'w') as f:
                json_results = {}
                for model_name, result in all_results.items():
                    json_results[model_name] = {
                        'type': result['type'],
                        'test_metrics': {k: float(v) for k, v in result['test_metrics'].items()},
                        'model_path': result.get('model_path', None)
                    }
                json.dump(json_results, f, indent=2)
            logger.info(f"\nResults saved to {results_file}")
    else:
        logger.warning("No models were successfully trained.")


if __name__ == "__main__":
    main()