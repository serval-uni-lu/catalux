import sys
import torch
import argparse
import numpy as np
from loguru import logger
from typing import Dict, List, Optional, Any, Union

from src.config import Config
from src.utils import seed_everything

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")


class FraudDetectionSystem:
    """Main interface for fraud detection operations."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the fraud detection system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config.from_yaml(config_path)
        seed_everything(self.config.seed)
        self.device = self.config.get_device()
        logger.info(f"Initialized system with device: {self.device}")
        
    def train(
        self,
        models: List[str],
        dataset: str = "etfd",
        model_type: str = "both",
        val_size: float = 0.1,
        test_size: float = 0.2,
        save_models: bool = True
    ) -> Dict[str, Any]:
        """Train fraud detection models.
        
        Args:
            models: List of model names to train (e.g., ['GCN', 'GAT', 'RealMLP'])
            dataset: Dataset name ('etfd', 'augmented', 'filtered')
            model_type: Type of models ('gnn', 'tabular', 'both')
            val_size: Validation split size
            test_size: Test split size
            save_models: Whether to save trained models
            
        Returns:
            Dictionary containing training results for each model
        """
        from src.train_all import train_all_models
        
        logger.info(f"Training {len(models)} models on {dataset} dataset")
        
        results = train_all_models(
            models=models,
            dataset=dataset,
            model_type=model_type,
            config=self.config,
            val_size=val_size,
            test_size=test_size,
            save_models=save_models
        )
        
        if results:
            best_model = max(results.items(), key=lambda x: x[1].get('test_metrics', {}).get('f1', 0))
            logger.info(f"Best model: {best_model[0]} with F1: {best_model[1]['test_metrics']['f1']:.4f}")
        
        return results
    
    def evaluate(
        self,
        model_names: Union[str, List[str]],
        dataset: str = "etfd",
        test_size: float = 0.2,
        models_dir: str = "models",
        save_results: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate trained models.
        
        Args:
            model_names: Model name(s) to evaluate
            dataset: Dataset name
            test_size: Test split size
            models_dir: Directory containing model files
            save_results: Optional path to save results CSV
            
        Returns:
            Dictionary containing evaluation metrics for each model
        """
        from src.evaluate import evaluate_model_by_name, save_evaluation_results
        
        if isinstance(model_names, str):
            model_names = [model_names]
        
        logger.info(f"Evaluating {len(model_names)} models on {dataset} dataset")
        
        all_results = {}
        for model_name in model_names:
            logger.info(f"Evaluating {model_name}...")
            try:
                metrics = evaluate_model_by_name(
                    model_name=model_name,
                    dataset=dataset,
                    config=self.config,
                    test_size=test_size,
                    models_dir=models_dir
                )
                all_results[model_name] = metrics
                logger.info(f"{model_name} - F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}
        
        if save_results:
            save_evaluation_results(all_results, save_results)
            logger.info(f"Results saved to {save_results}")
        
        return all_results
    
    def attack(
        self,
        model: Union[str, torch.nn.Module],
        dataset: str = "etfd",
        target_node: int = None,
        save_results: Optional[str] = None,
        **attack_params
    ) -> Dict[str, Any]:
        """Run adversarial attack on a model.
        
        Args:
            model: Model name or loaded model object
            dataset: Dataset name
            target_node: Target node ID for attack
            save_results: Optional path to save attack results JSON
            **attack_params: Additional attack parameters including:
                - evading_ids: List of target node IDs
                - main_initial_balance: Initial balance for main account
                - sybil_initial_balance: Initial balance for sybil accounts
                - max_balance_prop: Maximum proportion of balance to use
                - remove_exit: Whether to remove exit transactions
                - p_detection_threshold: Detection probability threshold
                - p_evasion_threshold: Evasion probability threshold
                - num_accounts: Number of accounts to attack if evading_ids not specified
                - num_steps: Number of attack steps
                - num_optim_steps: Number of optimization steps per attack step
                - gas_penalty: Gas penalty factor
            
        Returns:
            Dictionary containing attack results
        """
        from src.attack_interface import run_attack
        
        # log attack config
        if 'evading_ids' in attack_params and attack_params['evading_ids']:
            logger.info(f"Running attack on {len(attack_params['evading_ids'])} specified targets")
        elif target_node:
            logger.info(f"Running attack on target node {target_node}")
        else:
            logger.info(f"Running attack - will auto-detect targets with probability > {attack_params.get('p_detection_threshold', 0.5)}")
        
        # run the attack with all parameters
        results = run_attack(
            model=model,
            dataset=dataset,
            target_node=target_node,
            config=self.config,
            save_results=save_results,
            **attack_params
        )
        
        # log results summary
        if isinstance(results, dict):
            if 'individual_results' in results:
                # multiple targets
                success_rate = results.get('success_rate', 0.0)
                total = results.get('total_targets', 0)
                successful = results.get('successful_attacks', 0)
                logger.info(f"Attack completed: {successful}/{total} successful (success rate: {success_rate:.2%})")
            elif results.get('success'):
                # single target
                logger.info(f"Attack successful - Steps: {results.get('steps_taken', 'N/A')}, "
                           f"Final prob: {results.get('final_prob', 'N/A')}")
            else:
                logger.info("Attack failed to achieve target")
        
        
        return results
    
    def predict(
        self,
        model_name: str,
        node_ids: Union[int, List[int]],
        dataset: str = "etfd",
        models_dir: str = "models"
    ) -> np.ndarray:
        """Get predictions for specific nodes.
        
        Args:
            model_name: Name of the model
            node_ids: Node IDs to predict
            dataset: Dataset name
            models_dir: Directory containing model files
            
        Returns:
            Array of fraud probabilities
        """
        from src.predict import predict_nodes_by_name
        
        if isinstance(node_ids, int):
            node_ids = [node_ids]
        
        logger.info(f"Predicting {len(node_ids)} nodes using {model_name}")
        
        predictions = predict_nodes_by_name(
            model_name=model_name,
            node_ids=node_ids,
            dataset=dataset,
            config=self.config,
            models_dir=models_dir
        )
        
        return predictions
    
    def analyze_dataset(self, dataset: str = "etfd") -> Dict[str, Any]:
        """Analyze dataset statistics.
        
        Args:
            dataset: Dataset name
            
        Returns:
            Dictionary containing dataset statistics
        """
        from src.dataloader import load_txs, to_account_features
        
        logger.info(f"Analyzing {dataset} dataset")
        
        txs = load_txs(dataset)
        node_features, edges, edge_features = to_account_features(
            txs=txs, use_address=True, scam_features=True, edge_features=True
        )
        
        stats = {
            'num_transactions': len(txs),
            'num_nodes': len(node_features),
            'num_edges': len(edges),
            'fraud_rate': node_features['scam'].mean(),
            'avg_degree': len(edges) * 2 / len(node_features),
            'feature_dims': len([col for col in node_features.columns 
                               if col not in ['node_id', 'scam', 'address', 'scam_category']])
        }
        
        logger.info(f"Dataset stats: {stats['num_nodes']} nodes, "
                   f"{stats['num_edges']} edges, "
                   f"{stats['fraud_rate']:.2%} fraud rate")
        
        return stats


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # train command
    train_parser = subparsers.add_parser('train', help='Train scam detection models')
    train_parser.add_argument('--models', nargs='+', required=True,
                            help='Models to train (e.g., GCN GAT RealMLP)')
    train_parser.add_argument('--dataset', default='etfd',
                            help='Dataset name (etfd/augmented/filtered)')
    train_parser.add_argument('--type', choices=['gnn', 'tabular', 'both'], default='both',
                            help='Model type to train')
    train_parser.add_argument('--val-size', type=float, default=0.1,
                            help='Validation split size')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                            help='Test split size')
    train_parser.add_argument('--config', default='config.yaml',
                            help='Config file path')
    
    # evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    eval_parser.add_argument('--models', nargs='+', required=True,
                           help='Model names to evaluate (e.g., GCN GAT RealMLP)')
    eval_parser.add_argument('--dataset', default='etfd',
                           help='Dataset name')
    eval_parser.add_argument('--test-size', type=float, default=0.2,
                           help='Test split size')
    eval_parser.add_argument('--models-dir', default='models',
                           help='Directory containing model files')
    eval_parser.add_argument('--save-results', type=str,
                           help='Path to save results CSV (e.g., results/performance_metrics.csv)')
    eval_parser.add_argument('--config', default='config.yaml',
                           help='Config file path')
    
    # attack command
    attack_parser = subparsers.add_parser('attack', help='Run adversarial attack')
    attack_parser.add_argument('--model', 
                             help='Single model name or path (deprecated, use --model)')
    attack_parser.add_argument('--dataset', default='etfd',
                             help='Dataset name')
    attack_parser.add_argument('--evading-ids', nargs='+', type=int,
                             help='Target node IDs to attack')
    attack_parser.add_argument('--target-node', type=int,
                             help='Single target node ID (deprecated, use --evading-ids)')
    attack_parser.add_argument('--main-initial-balance', type=float, default=1e17,
                             help='Initial balance for main account (default: 1e17)')
    attack_parser.add_argument('--sybil-initial-balance', type=float, default=1e17,
                             help='Initial balance for sybil accounts (default: 1e17)')
    attack_parser.add_argument('--max-balance-prop', type=float, default=0.8,
                             help='Maximum proportion of balance to use (default: 0.8)')
    attack_parser.add_argument('--remove-exit', action='store_true', default=True,
                             help='Remove exit transactions (default: True)')
    attack_parser.add_argument('--no-remove-exit', dest='remove_exit', action='store_false',
                             help='Do not remove exit transactions')
    attack_parser.add_argument('--p-detection-threshold', type=float, default=0.5,
                             help='Detection probability threshold (default: 0.5)')
    attack_parser.add_argument('--num-accounts', type=int,
                             help='Number of accounts to attack if evading-ids not specified')
    attack_parser.add_argument('--num-steps', type=int, default=5,
                             help='Number of attack steps (default: 5)')
    attack_parser.add_argument('--num-optim-steps', type=int, default=10,
                             help='Number of optimization steps per attack step (default: 10)')
    attack_parser.add_argument('--p-evasion-threshold', type=float, default=0.5,
                             help='Evasion probability threshold (default: 0.5)')
    attack_parser.add_argument('--gas-penalty', type=float, default=0.0,
                             help='Gas penalty factor (default: 0.0)')
    attack_parser.add_argument('--save-results', type=str,
                             help='Path to save attack results JSON (e.g., results/GAT_attack.json)')
    attack_parser.add_argument('--type', default='evasion',
                             choices=['evasion', 'poisoning'],
                             help='Attack type')
    attack_parser.add_argument('--config', default='config.yaml',
                             help='Config file path')
    
    # predict command
    predict_parser = subparsers.add_parser('predict', help='Get predictions for nodes')
    predict_parser.add_argument('--model', required=True,
                              help='Model name')
    predict_parser.add_argument('--node-ids', nargs='+', type=int, required=True,
                              help='Node IDs to predict')
    predict_parser.add_argument('--dataset', default='etfd',
                              help='Dataset name')
    predict_parser.add_argument('--models-dir', default='models',
                              help='Directory containing model files')
    predict_parser.add_argument('--config', default='config.yaml',
                              help='Config file path')
    
    # analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset')
    analyze_parser.add_argument('--dataset', default='etfd',
                              help='Dataset name')
    analyze_parser.add_argument('--config', default='config.yaml',
                              help='Config file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # initialize system
    config_path = getattr(args, 'config', 'config.yaml')
    system = FraudDetectionSystem(config_path)
    
    # execute command
    if args.command == 'train':
        results = system.train(
            models=args.models,
            dataset=args.dataset,
            model_type=args.type,
            val_size=args.val_size,
            test_size=args.test_size
        )
        
    elif args.command == 'evaluate':
        results = system.evaluate(
            model_names=args.models,
            dataset=args.dataset,
            test_size=args.test_size,
            models_dir=args.models_dir,
            save_results=args.save_results
        )
        
        # print summary
        print("\nEvaluation Results:")
        print("-" * 60)
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                print(f"{model_name}: F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
            else:
                print(f"{model_name}: Error - {metrics['error']}")
        
    elif args.command == 'attack':
        models = args.models if args.models else ([args.model] if args.model else ['GCN'])
        evading_ids = args.evading_ids if args.evading_ids else ([args.target_node] if args.target_node else None)
        
        results = system.attack(
            model=models[0] if len(models) == 1 else models,
            dataset=args.dataset,
            target_node=evading_ids[0] if evading_ids and len(evading_ids) == 1 else None,
            models=models,
            evading_ids=evading_ids,
            main_initial_balance=args.main_initial_balance,
            sybil_initial_balance=args.sybil_initial_balance,
            max_balance_prop=args.max_balance_prop,
            remove_exit=args.remove_exit,
            p_detection_threshold=args.p_detection_threshold,
            num_accounts=args.num_accounts,
            num_steps=args.num_steps,
            num_optim_steps=args.num_optim_steps,
            p_evasion_threshold=args.p_evasion_threshold,
            gas_penalty=args.gas_penalty,
            save_results=args.save_results,
            attack_type=args.type
        )
        
    elif args.command == 'predict':
        predictions = system.predict(
            model_name=args.model,
            node_ids=args.node_ids,
            dataset=args.dataset,
            models_dir=args.models_dir
        )
        for node_id, prob in zip(args.node_ids, predictions):
            print(f"Node {node_id}: {prob:.4f}")
            
    elif args.command == 'analyze':
        stats = system.analyze_dataset(dataset=args.dataset)
        for key, value in stats.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()