import numpy as np
from loguru import logger
from typing import List, Union, Optional
from sklearn.preprocessing import StandardScaler

from .config import Config
from .model_interface import ModelFactory
from .dataloader import load_txs, to_account_features, to_pyg_data


def predict_nodes_by_name(
    model_name: str,
    node_ids: Union[int, List[int]],
    dataset: str = "etfd",
    config: Optional[Config] = None,
    models_dir: str = "models"
) -> np.ndarray:
    """Get fraud predictions for specific nodes using model name.
    
    Args:
        model_name: Name of the model
        node_ids: Node ID(s) to predict
        dataset: Dataset name
        config: Configuration object
        models_dir: Directory containing model files
        
    Returns:
        Array of fraud probabilities
    """
    # determine model path based on model type
    if model_name == 'LightGBM':
        model_path = f"{models_dir}/{model_name}.lgb"
    else:
        model_path = f"{models_dir}/{model_name}.pth"
    
    return predict_nodes(model_path, node_ids, dataset, config)


def predict_nodes(
    model_path: str,
    node_ids: Union[int, List[int]],
    dataset: str = "etfd",
    config: Optional[Config] = None
) -> np.ndarray:
    """Get fraud predictions for specific nodes.
    
    Args:
        model_path: Path to saved model
        node_ids: Node ID(s) to predict
        dataset: Dataset name
        config: Configuration object
        
    Returns:
        Array of fraud probabilities
    """
    if config is None:
        config = Config()
    
    if isinstance(node_ids, int):
        node_ids = [node_ids]
    
    # get model name
    model_name = model_path.split('/')[-1].split('.')[0]
    
    # load and preprocess data
    logger.info(f"Loading dataset: {dataset}")
    txs = load_txs(dataset)
    node_features, edges, edge_features = to_account_features(
        txs=txs, use_address=True, scam_features=True, edge_features=True
    )
    
    # load model with correct dimensions
    model = ModelFactory.load_model_with_data(model_path, config, node_features, edge_features)
    
    # validate node IDs
    max_node_id = len(node_features) - 1
    invalid_ids = [nid for nid in node_ids if nid < 0 or nid > max_node_id]
    if invalid_ids:
        raise ValueError(f"Invalid node IDs: {invalid_ids}. Valid range: 0-{max_node_id}")
    
    # check if model is GNN or tabular
    is_gnn = model_name in ['GCN', 'GAT', 'GATv2', 'SAGE', 'Chebyshev']
    
    if is_gnn:
        # prepare graph data
        node_labels = node_features['scam'].values
        feature_names = [col for col in node_features.columns 
                        if col not in ['node_id', 'scam', 'address', 'scam_category']]
        
        node_scaler = StandardScaler()
        edge_scaler = StandardScaler()
        scaled_node_features = node_scaler.fit_transform(node_features[feature_names].values)
        scaled_edge_features = edge_scaler.fit_transform(edge_features.values) if edge_features is not None else None
        
        graph = to_pyg_data(
            scaled_node_features, node_labels, edges.values,
            scaled_edge_features, config.get_device()
        )
        
        # get predictions
        predictions = model.predict(graph, node_ids)
    else:
        # tabular model
        feature_names = [col for col in node_features.columns 
                        if col not in ['node_id', 'scam', 'address', 'scam_category']]
        
        # scale features
        scaler = StandardScaler()
        all_features = scaler.fit_transform(node_features[feature_names].values)
        
        # get features for specific nodes
        node_features_subset = all_features[node_ids]
        
        # get predictions
        predictions = model.predict(node_features_subset)
    
    return predictions


def batch_predict(
    model_path: str,
    dataset: str = "etfd",
    batch_size: int = 1000,
    config: Optional[Config] = None
) -> np.ndarray:
    """Get predictions for all nodes in batches.
    
    Args:
        model_path: Path to saved model
        dataset: Dataset name
        batch_size: Batch size for prediction
        config: Configuration object
        
    Returns:
        Array of fraud probabilities for all nodes
    """
    if config is None:
        config = Config()
    
    # get model name
    model_name = model_path.split('/')[-1].split('.')[0]
    
    # load and preprocess data
    logger.info(f"Loading dataset: {dataset}")
    txs = load_txs(dataset)
    node_features, edges, edge_features = to_account_features(
        txs=txs, use_address=True, scam_features=True, edge_features=True
    )
    
    # load model with correct dimensions
    model = ModelFactory.load_model_with_data(model_path, config, node_features, edge_features)
    
    num_nodes = len(node_features)
    
    # check if model is GNN or tabular
    is_gnn = model_name in ['GCN', 'GAT', 'GATv2', 'SAGE', 'Chebyshev']
    
    if is_gnn:
        # for GNN, predict all at once (graph structure needed)
        node_labels = node_features['scam'].values
        feature_names = [col for col in node_features.columns 
                        if col not in ['node_id', 'scam', 'address', 'scam_category']]
        
        node_scaler = StandardScaler()
        edge_scaler = StandardScaler()
        scaled_node_features = node_scaler.fit_transform(node_features[feature_names].values)
        scaled_edge_features = edge_scaler.fit_transform(edge_features.values) if edge_features is not None else None
        
        graph = to_pyg_data(
            scaled_node_features, node_labels, edges.values,
            scaled_edge_features, config.get_device()
        )
        
        predictions = model.predict(graph)
    else:
        feature_names = [col for col in node_features.columns 
                        if col not in ['node_id', 'scam', 'address', 'scam_category']]
        
        # scale features
        scaler = StandardScaler()
        all_features = scaler.fit_transform(node_features[feature_names].values)
        
        # predict in batches
        predictions = []
        for i in range(0, num_nodes, batch_size):
            batch_features = all_features[i:min(i+batch_size, num_nodes)]
            batch_preds = model.predict(batch_features)
            predictions.append(batch_preds)
        
        predictions = np.concatenate(predictions)
    
    logger.info(f"Generated predictions for {len(predictions)} nodes")
    return predictions