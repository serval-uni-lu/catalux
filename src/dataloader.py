import torch
import numpy as np
import pandas as pd

from typing import Optional, Union
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

from .graph import Graph


def load_txs(dataset_name: str, filter: bool = True) -> pd.DataFrame:
    """Load transactions from a specified dataset.
    """
    if not dataset_name in ['ammari', 'etfd', 'agg']:
        raise ValueError("Dataset name must be either 'ammari' or 'etfd' or 'agg'.")

    txs = pd.read_csv(f'data/{dataset_name}/dataset.csv')
    txs_df = txs[['from_address', 'to_address', 'value', 'gas', 'gas_price', 'input', 'receipt_gas_used', 'from_scam', 'to_scam', 'from_category', 'to_category']].copy()
    txs_df = txs_df.dropna(subset=['to_address'])

    for col in ['value', 'gas', 'gas_price', 'receipt_gas_used']:
        txs_df[col] = pd.to_numeric(txs_df[col], errors='coerce')
    
    unique_addresses = pd.concat([txs_df['from_address'], txs_df['to_address']]).unique()
    address_to_id = {addr: idx for idx, addr in enumerate(unique_addresses)}
    txs_df['from_id'] = txs_df['from_address'].map(address_to_id)
    txs_df['to_id'] = txs_df['to_address'].map(address_to_id)
    txs_df['scam'] = txs_df['from_scam'] | txs_df['to_scam']
        
    return txs_df

def get_txs_for_ids(txs: pd.DataFrame, ids: Union[int, list]) -> pd.DataFrame:
    """Filter transactions for specific node IDs.
    """
    ids = [ids] if isinstance(ids, int) else ids
    txs_ids = txs[(txs['from_id'].isin(ids)) | (txs['to_id'].isin(ids))]
    return txs_ids.reset_index(drop=True)

def get_balance_for_id(txs, node_id: int) -> float:
    """Get account balance from transaction history."""
    is_sender = txs['from_id'] == node_id
    is_receiver = txs['to_id'] == node_id
    
    if not (is_sender.any() or is_receiver.any()):
        return 0.0  # no transactions for this node
    
    inflow = txs.loc[is_receiver, 'value'].sum() if is_receiver.any() else 0.0
    outflow = txs.loc[is_sender, 'value'].sum() if is_sender.any() else 0.0
    
    return float(inflow - outflow)

def get_scam_status_and_category(txs):
    """Extracts scam status and category for each node from transaction data.
    """
    from_info = txs[['from_id', 'from_scam', 'from_category']].rename(
        columns={'from_id': 'node_id', 'from_scam': 'scam', 'from_category': 'scam_category'}
    )
    to_info = txs[['to_id', 'to_scam', 'to_category']].rename(
        columns={'to_id': 'node_id', 'to_scam': 'scam', 'to_category': 'scam_category'}
    )
    all_info = pd.concat([from_info, to_info]).drop_duplicates(subset=['node_id'])
    all_info['scam'] = all_info['scam'].fillna(0).astype(int)
    all_info['scam_category'] = all_info['scam_category'].fillna('None')

    return all_info
    
def to_account_features(txs: pd.DataFrame, use_address: bool = False, scam_features: bool = False, edge_features: bool = False) -> tuple:
    """Engineers node features from a transaction DataFrame.
    """
    df = txs.copy()
    
    # --- preprocessing ---
    all_ids = pd.unique(df[['from_id', 'to_id']].values.ravel('K'))
    nodes_df = pd.DataFrame({'node_id': all_ids})
    if use_address:
        all_addresses = pd.unique(df[['from_address', 'to_address']].values.ravel('K'))
        nodes_df['address'] = all_addresses
    nodes_df.set_index('node_id', inplace=True)
    
    # --- scam features ---
    if scam_features:
        scam_feats = get_scam_status_and_category(df)
        scam_feats.set_index('node_id', inplace=True)
        nodes_df = nodes_df.join(scam_feats, how='left')
    
    # --- outgoing transaction features ---
    out = df.groupby('from_id', observed=True).agg({
        'value': ['count', 'sum', 'mean', 'std', 'min', 'max', 'median'],
        'gas': ['mean', 'min', 'max', 'median'],
        'gas_price': ['mean', 'min', 'max', 'median'],
        'receipt_gas_used': ['mean', 'sum', 'min', 'max', 'median'],
    })
    out.columns = [f'out_{col[0]}_{col[1]}' for col in out.columns]
    
    # --- incoming transaction features ---
    inc = df.groupby('to_id', observed=True).agg({
        'value': ['count', 'sum', 'mean', 'std', 'min', 'max', 'median'],
        'gas': ['mean', 'min', 'max', 'median'],
        'gas_price': ['mean', 'min', 'max', 'median'],
        'receipt_gas_used': ['mean', 'sum', 'min', 'max', 'median'],
    })
    inc.columns = [f'in_{col[0]}_{col[1]}' for col in inc.columns]
    
    # --- join outgoing and incoming ---
    nodes_df = nodes_df.join([out, inc], how='left').fillna(0)
    
    # --- aggregate features ---
    nodes_df['balance'] = nodes_df['in_value_sum'] - nodes_df['out_value_sum']
    nodes_df['n_tx_total'] = nodes_df['out_value_count'] + nodes_df['in_value_count']

    # --- unique counterparties ---
    nodes_df['n_unique_addr_in'] = df.groupby('to_id')['from_id'].nunique().reindex(nodes_df.index, fill_value=0)
    nodes_df['n_unique_addr_out'] = df.groupby('from_id')['to_id'].nunique().reindex(nodes_df.index, fill_value=0)
    
    # --- behavioral features ---
    nodes_df['prop_in_tx'] = nodes_df['in_value_count'] / nodes_df['n_tx_total']
    nodes_df['balance_ignoring_max_out'] = nodes_df['balance'] + nodes_df['out_value_max']
    nodes_df['is_sink_node'] = ((nodes_df['out_value_count'] == 0) & (nodes_df['in_value_count'] > 0)).astype(int)
    nodes_df['n_zero_value_tx_out'] = df[df['value'] == 0].groupby('from_id').size().reindex(nodes_df.index, fill_value=0)
    nodes_df['n_contract_calls'] = df[df['input'].str.len() > 2].groupby('from_id').size().reindex(nodes_df.index, fill_value=0)
    nodes_df['prop_contract_interaction'] = (nodes_df['n_contract_calls'] / nodes_df['n_tx_total'])
    
    nodes_df.reset_index(drop=False, inplace=True)

    # --- edges and edge features ---
    if not edge_features:
        return nodes_df
    else:
        edges = df[['from_id', 'to_id']].copy()
        edge_features = df[['value', 'gas', 'gas_price', 'receipt_gas_used']].copy()
        edge_features['total_cost'] = edge_features['gas'] * edge_features['gas_price']
        edge_features['gas_efficiency'] = edge_features['receipt_gas_used'] / edge_features['gas']

        return nodes_df, edges, edge_features

def filter_inactive_accounts(node_features, edges, edge_features):
    """Filter out inactive accounts based on transaction counts.
    """
    node_features = node_features[(node_features['n_tx_total'] > 1) & (node_features['in_value_count'] > 0)]

    valid_node_ids = set(node_features['node_id'])
    mask = edges['from_id'].isin(valid_node_ids) & edges['to_id'].isin(valid_node_ids)
    
    edges = edges[mask].reset_index(drop=True)
    edge_features = edge_features.loc[mask].copy().reset_index(drop=True)
    
    return node_features, edges, edge_features

def remap_ids(txs, node_features, edges):
    """Remap node and edge IDs to a continuous range starting from 0.
    """
    id_map = {old: new for new, old in enumerate(node_features['node_id'])}
    
    # txs
    txs['from_id'] = txs['from_id'].map(id_map)
    txs['to_id'] = txs['to_id'].map(id_map)
    
    # assign new ids to nans in from_id and to_id starting from the max id
    max_id = max(id_map.values())
    for col in ['from_id', 'to_id']:
        nan_mask = txs[col].isna()
        txs.loc[nan_mask, col] = (max_id + 1 + np.arange(nan_mask.sum()))
        txs[col] = txs[col].astype(int)
    
    # edges
    edges['from_id'] = edges['from_id'].map(id_map)
    edges['to_id'] = edges['to_id'].map(id_map)
    
    # node_features['node_id'] column should take values of index
    node_features.loc[:, 'node_id'] = node_features.index
    
    return txs, node_features, edges

def filter_and_remap(txs, node_features, edges, edge_features):
    """Filter inactive accounts and remap ids.
    """
    node_features, edges, edge_features = filter_inactive_accounts(node_features, edges, edge_features)
    txs, node_features, edges = remap_ids(txs, node_features, edges)

    return txs, node_features, edges, edge_features

def to_graph(
    node_features: np.ndarray,
    node_labels: np.ndarray,
    edges: np.ndarray,
    edge_features: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None
) -> Data:
    """Convert node and edge features into a PyTorch Geometric Data object with device support.
    """
    if device is None:
        device = torch.device('cpu')
    
    x = torch.tensor(node_features, dtype=torch.float, device=device)
    y = torch.tensor(node_labels, dtype=torch.long, device=device)
    edge_index = torch.tensor(edges.T, dtype=torch.long, device=device)

    edge_attr = None
    if edge_features is not None:
        edge_attr = torch.tensor(edge_features, dtype=torch.float, device=device)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return Graph(data, requires_grad=True)

class DataPreprocessor:
    """Loads and preprocesses transaction data for graph neural networks."""
    
    def __init__(self, dataset_name):
        self.txs = load_txs(dataset_name, False)
        self.node_features, self.edges, self.edge_features = to_account_features(self.txs, True, True, True)
        
        # self.txs, self.node_features, self.edges, self.edge_features = filter_and_remap(
        #     txs, node_features, edges, edge_features
        # )
        
        self.node_labels = self.node_features['scam'].astype(int)
        self.feature_names = [col for col in self.node_features.columns 
                             if col not in ['node_id', 'scam', 'address', 'scam_category']]
        
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        
        scaled_nodes = self.node_scaler.fit_transform(self.node_features[self.feature_names].values)
        scaled_edges = self.edge_scaler.fit_transform(self.edge_features.values)
        
        self.graph = to_graph(
            scaled_nodes,
            self.node_labels.values, 
            self.edges.values, 
            scaled_edges,
            device=torch.device('cpu')
        )
        print(f"Dataset preprocessing completed: {len(self.node_labels)} nodes, {len(self.edge_features)} edges")
        
    def normalize_node_features(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Normalize node features using the fitted scaler."""
        if isinstance(features, pd.DataFrame):
            features = features[self.feature_names].values
        return self.node_scaler.transform(features)

    def normalize_edge_features(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Normalize edge features using the fitted scaler."""
        if isinstance(features, pd.DataFrame):
            features = features.values
        return self.edge_scaler.transform(features)

    def unnormalize_node_features(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Unnormalize node features using the fitted scaler."""
        if isinstance(features, pd.DataFrame):
            features = features[self.feature_names].values
        return self.node_scaler.inverse_transform(features)

    def unnormalize_edge_features(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Unnormalize edge features using the fitted scaler."""
        if isinstance(features, pd.DataFrame):
            features = features.values
        return self.edge_scaler.inverse_transform(features)
