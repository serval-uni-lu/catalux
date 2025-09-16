import json
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from src.config import Config
from src.dataloader import load_txs, DataPreprocessor, get_balance_for_id
from src.train import GNNTrainer
from src.models import load_model
from src.predict import predict_for_ids, evaluate_model, evaluate_all_models
from src.attack import MultiTargetConstrainedGraphAttack, AttackConfig


def train(models: List[str], dataset: str = 'mtcga', config_path: Optional[str] = 'config.yaml'):
    config = Config.from_yaml(config_path) if config_path else Config()
    
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    print(f"Loading dataset: {dataset}")
    txs = load_txs(dataset)
    data = DataPreprocessor(txs)
    
    trainer = GNNTrainer(config)
    
    available_models = ['GCN', 'GAT', 'GATv2', 'SAGE', 'Chebyshev']
    if 'all' in models:
        models = available_models
    
    for model_name in models:
        if model_name not in available_models:
            print(f"Model {model_name} not available. Available: {available_models}")
            continue
        
        print(f"\nTraining {model_name}...")
        try:
            trainer.train_model(model_name, data.graph)
            print(f"Training completed for {model_name}")
        except Exception as e:
            print(f"Error training {model_name}: {e}")

def evaluate(models: Optional[List[str]] = None, dataset: str = 'mtcga', 
            split: str = 'test', config_path: Optional[str] = 'config.yaml'):
    config = Config.from_yaml(config_path) if config_path else Config()
    
    print(f"Loading dataset: {dataset}")
    txs = load_txs(dataset)
    data = DataPreprocessor(txs)
    
    if models is None:
        results = evaluate_all_models(data.graph, config, split)
    else:
        results = {}
        for model_name in models:
            try:
                metrics = evaluate_model(model_name, data.graph, config, split)
                results[model_name] = metrics
                print(f"{model_name} - {split} F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
    
    print("\n" + "="*50)
    print("Summary:")
    for model_name, metrics in results.items():
        if 'error' not in metrics:
            print(f"{model_name:10} - F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    
    return results

def attack(
        models: Optional[List[str]] = None, dataset: str = 'mtcga',
        split: str = 'test', config_path: Optional[str] = 'config.yaml',
        limit: Optional[int] = None):
    """Attack all detected scam accounts in the specified dataset and split.

    Args:
        models: List of model names to attack (None for all)
        dataset: Dataset name
        split: Data split to attack ('train', 'val', or 'test')
        config_path: Path to config file
        limit: Optional limit on number of nodes to attack per model
    """
    
    config = Config.from_yaml(config_path) if config_path else Config()
    
    print(f"Loading dataset: {dataset}")
    txs = load_txs(dataset)
    data = DataPreprocessor(txs)

    # load attack configuration
    attack_config = AttackConfig()

    # create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # get available models
    available_models = ['GCN', 'GAT', 'GATv2', 'SAGE', 'Chebyshev']
    if models is None or 'all' in models:
        models = available_models
    
    # load split indices
    for model_name in models:
        if model_name not in available_models:
            print(f"Model {model_name} not available. Available: {available_models}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Attacking {model_name} on {split} split")
        print('='*60)
        
        # load model and split
        model_dir = Path(f"models/{model_name}")
        if not model_dir.exists():
            print(f"Model {model_name} not found in models directory")
            continue
        
        split_file = model_dir / f"{split}.npy"
        if not split_file.exists():
            print(f"Split file {split_file} not found")
            continue
        
        try:
            # load model
            edge_dim = data.graph.edge_attr.shape[1] if data.graph.edge_attr is not None else 0
            model = load_model(model_name, data.graph.x.shape[1], edge_dim, config)
            device = config.get_device()
            model = model.to(device)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
        
        try:
            # get split indices
            split_indices = np.load(split_file)
            
            # get predictions for all nodes in split
            probs = predict_for_ids(model, data.graph, split_indices)
        except Exception as e:
            print(f"Error getting predictions for {model_name}: {e}")
            continue
        
        # get scam nodes
        scam_mask = probs >= 0.5
        scam_indices = split_indices[scam_mask]
        scam_probs = probs[scam_mask]
        
        if len(scam_indices) == 0:
            print(f"No scam accounts detected by {model_name} in {split} split")
            results = {
                'model': model_name,
                'split': split,
                'total_nodes': len(split_indices),
                'detected_scam_nodes': 0,
                'attacks': []
            }
        else:
            print(f"Found {len(scam_indices)} scam accounts to attack")
            
            attack_results = []
            successful_attacks = 0
            
            # attack each detected scam node
            attack_limit = limit if limit is not None else len(scam_indices)
            attacked_nodes = 0
            for idx, node_id in enumerate(scam_indices):
                if attacked_nodes >= attack_limit:
                    break
                    
                print(f"\nChecking node {node_id} ({idx+1}/{len(scam_indices)})")
                
                # check if node has positive balance
                node_balance = get_balance_for_id(txs, int(node_id))
                if node_balance <= 0:
                    print(f"  Skipping node {node_id} - zero or negative balance ({node_balance/1e18:.6f} ETH)")
                    continue
                
                print(f"  Node balance: {node_balance/1e18:.6f} ETH - proceeding with attack")
                attacked_nodes += 1
                
                try:
                    # run attack
                    attacker = MultiTargetConstrainedGraphAttack(
                        evading_ids=int(node_id),
                        model=model,
                        datapreprocessor=data,
                        config=attack_config
                    )
                    
                    result = attacker.run()
                    
                    # store result
                    attack_result = {
                        'node_id': int(node_id),
                        'success': bool(result.success),
                        'initial_prob': float(result.initial_prob),
                        'final_prob': float(result.final_prob),
                        'steps_taken': int(result.steps_taken),
                        'sybils_created': int(result.sybils_created),
                        'budget_spent': float(result.budget_spent),
                        'budget_spent_prop': float(result.budget_spent_prop),
                        'total_budget': float(result.total_budget),
                        'budget_exhausted': bool(result.budget_exhausted),
                        'early_stopped': bool(result.early_stopped),
                        'num_transactions': len(result.transactions),
                        'transactions': [dict(tx) for tx in result.transactions],
                        'probabilities': [
                            {
                                'step': int(prob['step']),
                                'probs': {str(k): float(v) for k, v in prob['probs'].items()}
                            } for prob in result.probabilities
                        ]
                    }
                    
                    initial_prob = float(result.initial_prob)
                    if result.success:
                        successful_attacks += 1
                        print(f"    Attack successful! P(fraud): {initial_prob:.3f} → {result.final_prob:.3f}")
                        print(f"    Budget: {result.budget_spent/1e18:.3f}/{result.total_budget/1e18:.3f} ETH ({result.budget_spent_prop:.1%})")
                        print(f"    Sybils: {result.sybils_created}, Steps: {result.steps_taken}")
                        if result.steps_taken == 0:
                            print(f"    Method: No transformations needed (already below threshold)")
                    else:
                        print(f"    Attack failed. P(fraud): {initial_prob:.3f} → {result.final_prob:.3f}")
                        if result.budget_exhausted:
                            print(f"    Reason: Budget exhausted ({result.budget_spent:.0f}/{result.total_budget:.0f})")
                        else:
                            print(f"    Reason: No beneficial transformations found")
                    
                    attack_results.append(attack_result)
                    
                except Exception as e:
                    print(f"  Error attacking node {node_id}: {e}")
                    attack_results.append({
                        'node_id': int(node_id),
                        'error': str(e),
                        'success': bool(False)
                    })
            
            # compute additional statistics first
            if attack_results:
                successful_results = [r for r in attack_results if r.get('success', False)]
                avg_budget_used = float(np.mean([r['budget_spent_prop'] for r in successful_results])) if successful_results else 0.0
                avg_sybils = float(np.mean([r['sybils_created'] for r in successful_results])) if successful_results else 0.0
                avg_steps = float(np.mean([r['steps_taken'] for r in successful_results])) if successful_results else 0.0
            else:
                avg_budget_used = avg_sybils = avg_steps = 0.0

            # compute statistics
            results = {
                'model': str(model_name),
                'split': str(split),
                'dataset': str(dataset),
                'total_nodes': int(len(split_indices)),
                'detected_scam_nodes': int(len(scam_indices)),
                'attacked_nodes': int(len(attack_results)),
                'successful_attacks': int(successful_attacks),
                'attack_success_rate': float(successful_attacks / len(attack_results) if len(attack_results) > 0 else 0.0),
                'avg_budget_used': float(avg_budget_used),
                'avg_sybils_created': float(avg_sybils),
                'avg_steps_taken': float(avg_steps),
                'attack_config': {
                    'max_budget_prop': float(attack_config.max_budget_prop),
                    'p_evasion_threshold': float(attack_config.p_evasion_threshold),
                    'max_transformations': int(attack_config.max_transformations),
                    'max_sybils': int(attack_config.max_sybils)
                },
                'attacks': attack_results
            }
            
            print(f"\n{'='*40}")
            print(f"Attack Summary for {model_name}:")
            print(f"  Total nodes in {split}: {len(split_indices)}")
            print(f"  Detected scam nodes: {len(scam_indices)}")
            print(f"  Attacked nodes: {len(attack_results)}")
            print(f"  Successful attacks: {successful_attacks}/{len(attack_results)}")
            print(f"  Success rate: {results['attack_success_rate']:.2%}")
            if successful_results:
                print(f"  Avg budget used: {avg_budget_used:.1%}")
                print(f"  Avg sybils created: {avg_sybils:.1f}")
                print(f"  Avg steps taken: {avg_steps:.1f}")
        
        # save results to JSON
        output_file = results_dir / f"{model_name}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

def analyze(results_dir: str = 'results') -> pd.DataFrame:
    """Analyze attack results from all models and create summary dataframe.

    Args:
        results_dir: Directory containing attack result JSON files

    Returns:
        DataFrame with comprehensive attack analysis
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory {results_dir} not found")
        return pd.DataFrame()

    all_results = []

    for json_file in results_path.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)

        model_name = data['model']
        attacks = data.get('attacks', [])

        # calculate statistics
        successful_attacks = [a for a in attacks if a.get('success', False)]
        num_successful = len(successful_attacks)
        num_total = len(attacks)

        if successful_attacks:
            avg_budget_prop = np.mean([a['budget_spent_prop'] for a in successful_attacks])
            avg_sybils = np.mean([a['sybils_created'] for a in successful_attacks])
            avg_steps = np.mean([a['steps_taken'] for a in successful_attacks])
            avg_prob_reduction = np.mean([
                a['initial_prob'] - a['final_prob'] for a in successful_attacks
            ])
        else:
            avg_budget_prop = avg_sybils = avg_steps = avg_prob_reduction = 0

        all_results.append({
            'model': model_name,
            'split': data['split'],
            'dataset': data['dataset'],
            'total_nodes': data['total_nodes'],
            'detected_scam': data['detected_scam_nodes'],
            'attacked_nodes': num_total,
            'successful_attacks': num_successful,
            'attack_success_rate': num_successful / num_total if num_total > 0 else 0,
            'avg_budget_prop': avg_budget_prop,
            'avg_sybils': avg_sybils,
            'avg_steps': avg_steps,
            'avg_prob_reduction': avg_prob_reduction
        })

    df = pd.DataFrame(all_results)

    if not df.empty:
        print("\n" + "="*60)
        print("Attack Analysis Summary")
        print("="*60)
        print(df.to_string(index=False))

        # save to CSV
        csv_path = results_path / "attack_analysis.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nAnalysis saved to {csv_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models for fraud detection")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--models', nargs='+', default=['all'], 
                            help='Models to train (GCN, GAT, GATv2, SAGE, Chebyshev, or all)')
    train_parser.add_argument('--dataset', default='mtcga', help='Dataset to use')
    train_parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.add_argument('--models', nargs='*', default=None,
                           help='Models to evaluate (leave empty for all)')
    eval_parser.add_argument('--dataset', default='mtcga', help='Dataset to use')
    eval_parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                           help='Split to evaluate on')
    eval_parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    attack_parser = subparsers.add_parser('attack', help='Attack detected scam accounts')
    attack_parser.add_argument('--models', nargs='*', default=None,
                           help='Models to attack (leave empty for all)')
    attack_parser.add_argument('--dataset', default='mtcga', help='Dataset to use')
    attack_parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                           help='Split to attack on')
    attack_parser.add_argument('--config', default='config.yaml', help='Config file path')
    attack_parser.add_argument('--limit', type=int, default=None,
                           help='Limit number of nodes to attack per model')

    analyze_parser = subparsers.add_parser('analyze', help='Analyze attack results')
    analyze_parser.add_argument('--results', default='results', help='Results directory')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args.models, args.dataset, args.config)
    elif args.command == 'evaluate':
        evaluate(args.models, args.dataset, args.split, args.config)
    elif args.command == 'attack':
        attack(args.models, args.dataset, args.split, args.config, args.limit)
    elif args.command == 'analyze':
        analyze(args.results)
    else:
        parser.print_help()