import os
import gc
import torch
import argparse
import warnings
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import Config
from src.attack import Attack
from src.dataloader import DataPreprocessor
from src.models import load_pretrained, predict_for_ids
from src.utils import seed_everything

os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

warnings.filterwarnings('ignore')

MODEL_NAMES = ['GCN', 'GAT', 'GATv2', 'SAGE', 'Chebyshev']

def main():
    parser = argparse.ArgumentParser(description="Run evasion attacks on selected models.")

    parser.add_argument('--models', type=str, nargs='+', default=MODEL_NAMES,
                        help=f"List of model names to use. Choices: {', '.join(MODEL_NAMES)}. Default: {MODEL_NAMES}.")
    parser.add_argument('--num_accounts', type=int, default=2,
                        help="Number of accounts. Default: 2.")
    parser.add_argument('--num_steps', type=int, default=2,
                        help="Number of steps. Default: 2.")
    parser.add_argument('--num_optim_steps', type=int, default=5,
                        help="Number of optimization steps. Default: 5.")
    parser.add_argument('--p_evasion_threshold', type=float, default=0.5,
                        help="Evasion threshold probability. Default: 0.5.")
    parser.add_argument('--gas_penalty', type=float, default=0.0,
                        help="Gas penalty. Default: 0.0.")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help="Path to configuration file. Default: config.yaml.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    
    seed_everything(config.seed)
    
    torch.set_float32_matmul_precision('high')
    
    device = config.get_device()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    n = args.num_accounts
    steps = args.num_steps
    optim_steps = args.num_optim_steps
    gas_penalty = args.gas_penalty
    p_evasion_threshold = args.p_evasion_threshold

    print("Loading and preprocessing data...")
    data = DataPreprocessor('etfd')
    labels = data.node_labels
    train_indices, val_indices = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=config.seed
    )
    scam_val_ids = val_indices[labels[val_indices] == 1] # scams in validation set

    num_features = len(data.feature_names)
    edge_dim = data.edge_features.shape[1]
    print(f"Dataset loaded: {len(data.node_labels)} nodes, {len(data.edge_features)} edges")

    model_names = args.models

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        try:
            model = load_pretrained(f'checkpoints_full/{model_name}.pth', num_features, edge_dim, config)

            pred_probas = predict_for_ids(model, data.graph, scam_val_ids)
            candidates_ids = np.where(pred_probas > 0.5)[0].tolist()
            actual_node_ids = [int(scam_val_ids[i]) for i in candidates_ids[:n]]
            
            print(f"Found {len(actual_node_ids)} candidate accounts for attack")

            results = []
            for evading_id in actual_node_ids:
                print('\n' + '=' * 40)
                print(f'Running {model_name} on id {evading_id}')
                print('=' * 40)

                attack = Attack(model, data, evading_id)
                result = attack.run(
                    num_steps=steps,
                    num_optim_steps=optim_steps,
                    p_evasion_threshold=p_evasion_threshold,
                    gas_penalty=gas_penalty
                )
                results.append(result)
                
                del attack
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue
        finally:
            if 'model' in locals():
                del model
            if device.type == 'cuda':
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
            gc.collect()
    
    print(f"\n{'='*60}")
    print("Attack analysis completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()