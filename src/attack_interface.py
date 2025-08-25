import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, Any

from tqdm import tqdm
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .config import Config
from .dataloader import DataPreprocessor, load_txs, to_account_features, get_txs_for_ids, get_balance_for_id
from .gnn_models import predict_for_ids
from .gnn_attack import GNNAttack
from .tab_attack import TabularAttack
from .model_interface import ModelFactory


def run_attack(
    model: Union[str, torch.nn.Module],
    dataset: str,
    target_node: Optional[int] = None,
    config: Optional[Config] = None,
    save_results: Optional[str] = None,
    **attack_params
) -> Dict[str, Any]:
    """Run adversarial attack on a fraud detection model with proper feature scaling.
    
    Args:
        model: Model name/path or loaded model object
        dataset: Dataset name
        target_node: Target node ID (None for random selection)
        config: Configuration object
        save_results: Optional path to save attack results JSON
        **attack_params: Additional attack parameters
        
    Returns:
        Dictionary containing attack results
    """
    if config is None:
        config = Config()
    
    # extract parameters
    evading_ids = attack_params.get('evading_ids', [target_node] if target_node else None)
    main_initial_balance = attack_params.get('main_initial_balance', 2e17)
    sybil_initial_balance = attack_params.get('sybil_initial_balance', 1e17)
    max_balance_prop = attack_params.get('max_balance_prop', 0.8)
    remove_exit = attack_params.get('remove_exit', True)
    p_detection_threshold = attack_params.get('p_detection_threshold', 0.5)
    p_evasion_threshold = attack_params.get('p_evasion_threshold', 0.5)
    num_accounts = attack_params.get('num_accounts', None)
    num_steps = attack_params.get('num_steps', 5)
    num_optim_steps = attack_params.get('num_optim_steps', 10)
    gas_penalty = attack_params.get('gas_penalty', 0.0)
    
    # load and preprocess data
    logger.info(f"Loading dataset: {dataset}")
    txs = load_txs(dataset)
    data_preprocessor = DataPreprocessor(txs)
    
    # load model if needed
    if isinstance(model, str):
        if '/' in model or '.' in model:
            # model is path
            model_obj = ModelFactory.load_model_with_data(
                model, config, 
                data_preprocessor.node_features, 
                data_preprocessor.edge_features
            )
            model_name = model.split('/')[-1].split('.')[0]
        else:
            # model is name
            model_name = model
            if model in ['RealMLP', 'TabM']:
                model_path = f"models/{model}.pth"
            elif model == 'LightGBM':
                model_path = f"models/{model}.lgb"
            else:
                model_path = f"models/{model}.pth"
            try:
                model_obj = ModelFactory.load_model_with_data(
                    model_path, config, 
                    data_preprocessor.node_features, 
                    data_preprocessor.edge_features
                )
            except Exception as e:
                logger.error(f"Could not load pretrained model from {model_path}: {e}")
                raise
    else:
        model_obj = model
        model_name = "custom"
    
    # determine if model is GNN or tabular
    is_gnn = model_name in ['GCN', 'GAT', 'GATv2', 'SAGE', 'Chebyshev']
    is_tabular = model_name in ['RealMLP', 'TabM', 'LightGBM']
    
    # prepare scaler for tabular models
    scaler = None
    if is_tabular:
        node_features_df = to_account_features(data_preprocessor.txs)
        feature_names = [col for col in node_features_df.columns 
                        if col not in ['node_id', 'scam', 'address', 'scam_category']]
        X = node_features_df[feature_names].values
        y = data_preprocessor.node_labels
        
        # split data exactly as training
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=config.seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.1/(1-0.2), 
            stratify=y_temp, random_state=config.seed
        )
        
        # create and fit scaler on training data only
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_scaled = scaler.transform(X)
    
    # if evading_ids not specified, find targets from test set
    if evading_ids is None:
        # create train/test split using same strategy as training
        node_indices = np.arange(len(data_preprocessor.node_labels))
        train_idx, test_idx = train_test_split(
            node_indices, test_size=0.2, stratify=data_preprocessor.node_labels, random_state=config.seed
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.1/(1-0.2), stratify=data_preprocessor.node_labels[train_idx], random_state=config.seed
        )
        
        # select fraudulent nodes from test set only
        test_fraud_mask = data_preprocessor.node_labels[test_idx] == 1
        fraud_nodes = test_idx[test_fraud_mask]
        
        if len(fraud_nodes) == 0:
            raise ValueError("No fraudulent nodes found in test set")
        
        # get predictions for fraud nodes from test set
        try:
            if is_tabular:
                # for tabular models, use scaled features
                if hasattr(model_obj.model, 'predict_proba'):
                    predictions = model_obj.model.predict_proba(X_scaled)[:, 1]
                elif hasattr(model_obj.model, 'forward'):
                    # pyTorch model
                    model_obj.model.eval()
                    with torch.no_grad():
                        x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=config.get_device())
                        logits = model_obj.model(x_tensor)
                        predictions = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                else:
                    raise ValueError(f"Cannot get predictions from {type(model_obj.model)}")
                fraud_probs = predictions[fraud_nodes]
            elif hasattr(model_obj, 'predict'):
                # use graph data from preprocessor for GNN models
                predictions = model_obj.predict(data_preprocessor.graph.data)
                fraud_probs = predictions[fraud_nodes]
            else:
                # for raw GNN models, use predict_for_ids
                fraud_probs = predict_for_ids(model_obj, data_preprocessor.graph.data, fraud_nodes)
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            # fallback for GNN models
            if not is_tabular:
                fraud_probs = predict_for_ids(model_obj, data_preprocessor.graph.data, fraud_nodes)
            else:
                raise
        
        # filter by detection threshold
        detected_mask = fraud_probs > p_detection_threshold
        detected_nodes = fraud_nodes[detected_mask]
        
        # limit number of accounts if specified
        if num_accounts is not None and len(detected_nodes) > num_accounts:
            # take top num_accounts
            evading_ids = detected_nodes.tolist()[:num_accounts]
        else:
            evading_ids = detected_nodes.tolist()
        
        logger.info(f"Selected {len(evading_ids)} target nodes from test set with probability > {p_detection_threshold}")
    
    # validate target nodes
    for node_id in evading_ids:
        if node_id >= len(data_preprocessor.node_labels):
            raise ValueError(f"Invalid target node {node_id}. Dataset has {len(data_preprocessor.node_labels)} nodes")
    
    # run attacks
    all_results = []

    for evading_id in tqdm(evading_ids, desc="Processing evading IDs"):
        # get initial detection probability for this specific target
        try:
            if is_tabular:
                # for tabular models, use scaled features
                if hasattr(model_obj.model, 'predict_proba'):
                    predictions = model_obj.model.predict_proba(X_scaled)[:, 1]
                elif hasattr(model_obj.model, 'forward'):
                    # pyTorch model
                    model_obj.model.eval()
                    with torch.no_grad():
                        x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=config.get_device())
                        logits = model_obj.model(x_tensor)
                        predictions = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                else:
                    predictions = None
                initial_detection_prob = float(predictions[evading_id]) if predictions is not None else None
            elif hasattr(model_obj, 'predict'):
                predictions = model_obj.predict(data_preprocessor.graph.data)
                initial_detection_prob = float(predictions[evading_id])
            else:
                initial_detection_prob = float(predict_for_ids(model_obj, data_preprocessor.graph.data, [evading_id])[0])
        except:
            initial_detection_prob = None
        
        # get initial transaction count for this account
        account_txs = get_txs_for_ids(data_preprocessor.txs, evading_id)
        initial_tx_count = len(account_txs)
        initial_in_count = len(account_txs[account_txs['to_id'] == evading_id])
        initial_out_count = len(account_txs[account_txs['from_id'] == evading_id])
        
        # get initial balance
        initial_balance = get_balance_for_id(account_txs, evading_id)
        
        # calculate initial transaction volume
        initial_in_volume = float(account_txs[account_txs['to_id'] == evading_id]['value'].sum())
        initial_out_volume = float(account_txs[account_txs['from_id'] == evading_id]['value'].sum())
        
        # record start time for attack performance
        attack_start_time = time.time()
        
        if is_gnn:
            result = run_gnn_evasion_attack(
                model_obj, data_preprocessor, evading_id, config,
                num_steps=num_steps,
                num_optim_steps=num_optim_steps,
                p_evasion_threshold=p_evasion_threshold,
                gas_penalty=gas_penalty,
                max_balance_prop=max_balance_prop,
                remove_exit=remove_exit,
                main_initial_balance=main_initial_balance,
                sybil_initial_balance=sybil_initial_balance
            )
        elif is_tabular:
            result = run_tabular_evasion_attack(
                model_obj, data_preprocessor, evading_id, config,
                scaler=scaler,
                num_steps=num_steps,
                num_optim_steps=num_optim_steps,
                p_evasion_threshold=p_evasion_threshold,
                gas_penalty=gas_penalty,
                max_balance_prop=max_balance_prop,
                remove_exit=remove_exit,
                main_initial_balance=main_initial_balance,
                sybil_initial_balance=sybil_initial_balance
            )
        else:
            logger.warning(f"Unknown model type for {model_name}, attempting GNN attack")
            result = run_gnn_evasion_attack(
                model_obj, data_preprocessor, evading_id, config,
                num_steps=num_steps,
                num_optim_steps=num_optim_steps,
                p_evasion_threshold=p_evasion_threshold,
                gas_penalty=gas_penalty,
                max_balance_prop=max_balance_prop,
                remove_exit=remove_exit,
                main_initial_balance=main_initial_balance,
                sybil_initial_balance=sybil_initial_balance
            )
        
        attack_duration = time.time() - attack_start_time
        
        # enhanced result information
        result.update({
            'evading_id': evading_id,
            'model_name': model_name,
            'dataset': dataset,
            
            # initial detection information
            'initial_detection_prob': initial_detection_prob,
            'detection_threshold_used': p_detection_threshold,
            'evasion_threshold_target': p_evasion_threshold,
            
            # initial account state
            'initial_tx_count': initial_tx_count,
            'initial_incoming_tx_count': initial_in_count,
            'initial_outgoing_tx_count': initial_out_count,
            'initial_balance': float(initial_balance),
            'initial_incoming_volume': initial_in_volume,
            'initial_outgoing_volume': initial_out_volume,
            'initial_net_flow': initial_in_volume - initial_out_volume,
            
            # attack performance metrics
            'attack_duration_seconds': float(attack_duration),
            'attack_config_used': {
                'main_initial_balance': main_initial_balance,
                'sybil_initial_balance': sybil_initial_balance,
                'max_balance_prop': max_balance_prop,
                'remove_exit': remove_exit,
                'num_steps': num_steps,
                'num_optim_steps': num_optim_steps,
                'gas_penalty': gas_penalty
            }
        })
        
        # add derived metrics
        if result.get('initial_prob') is not None and result.get('final_prob') is not None:
            result['probability_reduction'] = float(result['initial_prob'] - result['final_prob'])
            result['probability_reduction_percent'] = float((result['initial_prob'] - result['final_prob']) / result['initial_prob'] * 100) if result['initial_prob'] > 0 else 0
        
        if result.get('transactions'):
            result['avg_transaction_value'] = float(result['total_value_transferred'] / len(result['transactions'])) if result['transactions'] else 0
            result['avg_gas_cost_per_tx'] = float(result['total_gas_cost'] / len(result['transactions'])) if result['transactions'] else 0
        
        all_results.append(result)
    
    # prepare results
    if len(evading_ids) == 1:
        results = all_results[0]
    else:
        # return summary with all results and aggregate statistics
        successful_attacks = sum(1 for r in all_results if r.get('success', False))
        
        # calculate aggregate statistics
        total_attack_duration = sum(r.get('attack_duration_seconds', 0) for r in all_results)
        avg_attack_duration = total_attack_duration / len(all_results) if all_results else 0
        
        # aggregate initial account statistics
        valid_detection_probs = [r['initial_detection_prob'] for r in all_results if r.get('initial_detection_prob') is not None]
        avg_initial_detection_prob = sum(valid_detection_probs) / len(valid_detection_probs) if valid_detection_probs else None
        
        total_initial_txs = sum(r.get('initial_tx_count', 0) for r in all_results)
        avg_initial_txs = total_initial_txs / len(all_results) if all_results else 0
        
        total_initial_balance = sum(r.get('initial_balance', 0) for r in all_results)
        avg_initial_balance = total_initial_balance / len(all_results) if all_results else 0
        
        # aggregate attack performance
        successful_results = [r for r in all_results if r.get('success', False)]
        if successful_results:
            avg_steps_successful = sum(r.get('steps_taken', 0) for r in successful_results) / len(successful_results)
            avg_prob_reduction = sum(r.get('probability_reduction', 0) for r in successful_results) / len(successful_results)
            avg_transactions_generated = sum(len(r.get('transactions', [])) for r in successful_results) / len(successful_results)
            total_sybils_created = sum(r.get('sybils_created', 0) for r in successful_results)
        else:
            avg_steps_successful = 0
            avg_prob_reduction = 0
            avg_transactions_generated = 0
            total_sybils_created = sum(r.get('sybils_created', 0) for r in all_results)
        
        results = {
            # basic results
            'success': successful_attacks > 0,
            'total_targets': len(evading_ids),
            'successful_attacks': successful_attacks,
            'success_rate': successful_attacks / len(evading_ids) if evading_ids else 0.0,
            
            # aggregate target characteristics
            'avg_initial_detection_prob': avg_initial_detection_prob,
            'avg_initial_tx_count': avg_initial_txs,
            'avg_initial_balance': avg_initial_balance,
            'total_initial_tx_count': total_initial_txs,
            'total_initial_balance': total_initial_balance,
            
            # aggregate attack performance
            'total_attack_duration_seconds': total_attack_duration,
            'avg_attack_duration_seconds': avg_attack_duration,
            'avg_steps_taken_successful': avg_steps_successful,
            'avg_probability_reduction': avg_prob_reduction,
            'avg_transactions_generated': avg_transactions_generated,
            'total_sybils_created': total_sybils_created,
            
            # configuration used
            'attack_config': {
                'model_name': model_name,
                'dataset': dataset,
                'main_initial_balance': main_initial_balance,
                'sybil_initial_balance': sybil_initial_balance,
                'max_balance_prop': max_balance_prop,
                'remove_exit': remove_exit,
                'p_detection_threshold': p_detection_threshold,
                'num_accounts': num_accounts,
                'num_steps': num_steps,
                'num_optim_steps': num_optim_steps,
                'p_evasion_threshold': p_evasion_threshold,
                'gas_penalty': gas_penalty
            },
            
            # individual results
            'individual_results': all_results
        }
    
    # save results
    if save_results and results:
        save_path = Path(save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # save results
        with open(save_path, 'w') as f:
            json.dump(convert_numpy_types(results), f, indent=2, default=str)
        logger.info(f"Attack results saved to {save_path}")
    
    return results


def run_gnn_evasion_attack(
    model,
    data_preprocessor: DataPreprocessor,
    target_node: int,
    num_steps: int = 5,
    num_optim_steps: int = 10,
    p_evasion_threshold: float = 0.5,
    gas_penalty: float = 0.0,
    max_balance_prop: float = 0.8,
    remove_exit: bool = True,
    main_initial_balance: float = 1e17,
    sybil_initial_balance: float = 1e17,
) -> Dict[str, Any]:
    """Run evasion attack on GNN model."""
    
    # initialize attack
    attack = GNNAttack(
        model=model.model if hasattr(model, 'model') else model,
        datapreprocessor=data_preprocessor,
        evading_id=target_node,
        main_initial_balance=main_initial_balance,
        sybil_initial_balance=sybil_initial_balance,
        max_balance_prop=max_balance_prop,
        remove_exit=remove_exit
    )
    
    # run attack
    results = attack.run(
        num_steps=num_steps,
        num_optim_steps=num_optim_steps,
        p_evasion_threshold=p_evasion_threshold,
        gas_penalty=gas_penalty
    )
    
    return results


def run_tabular_evasion_attack(
    model,
    data_preprocessor: DataPreprocessor,
    target_node: int,
    config: Optional[Config] = None,
    scaler: Optional[StandardScaler] = None,
    num_steps: int = 5,
    num_optim_steps: int = 10,
    p_evasion_threshold: float = 0.5,
    gas_penalty: float = 0.0,
    max_balance_prop: float = 0.8,
    remove_exit: bool = False,
    main_initial_balance: float = 1e17,
    sybil_initial_balance: float = 1e17,
) -> Dict[str, Any]:
    """Run evasion attack on tabular model with proper feature scaling."""
    
    # determine model type
    if hasattr(model, 'model_type'):
        model_type = model.model_type
    elif hasattr(model, 'model') and hasattr(model.model, '__class__'):
        # check the wrapped model's class
        class_name = model.model.__class__.__name__
        if 'LightGBM' in class_name:
            model_type = 'lightgbm'
        else:
            model_type = 'pytorch'
    elif hasattr(model, '__class__'):
        class_name = model.__class__.__name__
        if 'LightGBM' in class_name:
            model_type = 'lightgbm'
        else:
            model_type = 'pytorch'
    else:
        model_type = 'pytorch'
    
    node_features_df = to_account_features(data_preprocessor.txs)
    feature_names = [col for col in node_features_df.columns 
                    if col not in ['node_id', 'scam', 'address', 'scam_category']]
    
    if scaler is None:
        X = node_features_df[feature_names].values
        y = data_preprocessor.node_labels
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=config.seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.1/(1-0.2), 
            stratify=y_temp, random_state=config.seed
        )
        
        scaler = StandardScaler()
        scaler.fit(X_train)
    
    # initialize attack
    attack = TabularAttack(
        model=model.model if hasattr(model, 'model') else model,
        model_type=model_type,
        feature_names=feature_names,
        scaler=scaler,
        evading_id=target_node,
        txs=data_preprocessor.txs,
        main_initial_balance=main_initial_balance,
        sybil_initial_balance=sybil_initial_balance,
        max_balance_prop=max_balance_prop,
        remove_exit=remove_exit
    )
    
    # run attack
    results = attack.run(
        num_steps=num_steps,
        num_optim_steps=num_optim_steps,
        p_evasion_threshold=p_evasion_threshold,
        gas_penalty=gas_penalty
    )
    
    return results
