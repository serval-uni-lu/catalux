import numpy as np
import pandas as pd
import networkx as nx

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

from sklearn.preprocessing import StandardScaler

def load_dataset(name):
    """Loads a dataset by name and returns the node features and edges.
    """
    if name == "ammari":
        return load_ammari()
    elif name == "etfd":
        return load_etfd()
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    
def load_ammari():
    """Loads the Ammari dataset and engineers node features.
    """
    path = 'data/ammari/dataset.csv'
    txs_df = pd.read_csv(path)
    to_drop = ['hash', 'nonce', 'transaction_index', 'input', 'block_hash']
    txs_df.drop(columns=to_drop, inplace=True, errors='ignore')
    features, edges = engineer_node_features(txs_df)
    return features, edges

def load_etfd():
    """Loads the ETFD dataset and engineers node features.
    """
    path = 'data/etfd/dataset.csv'
    txs_df = pd.read_csv(path)
    to_drop = ['hash', 'nonce', 'blockHash', 'transactionIndex', 'isError',
               'txreceipt_status', 'contractAddress', 'cumulativeGasUsed',
               'confirmations', 'methodId', 'functionName']
    txs_df.drop(columns=to_drop, inplace=True, errors='ignore')
    features, edges = engineer_node_features(txs_df)
    return features, edges

def engineer_node_features(base_df: pd.DataFrame) -> pd.DataFrame:
    """Engineers node features from a txs_df transaction DataFrame.
    """
    df = base_df.copy()

    # --- preprocessing ---
    numeric_cols = ['value', 'gas', 'gas_price', 'receipt_gas_used', 'block_number']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['from_address', 'to_address'], inplace=True)
    df['block_number'] = pd.to_numeric(df['block_number'], errors='coerce')

    all_addresses = pd.unique(df[['from_address', 'to_address']].values.ravel('K'))
    all_addresses = all_addresses[pd.notna(all_addresses)]
    
    nodes_df = pd.DataFrame({'address': all_addresses})
    nodes_df['node_id'] = range(len(nodes_df))
    nodes_df.set_index('address', inplace=True)

    # --- outgoing transaction features ---
    tx_outgoing_grouped = df.groupby('from_address')
    nodes_df['out_degree'] = tx_outgoing_grouped.size().reindex(nodes_df.index).fillna(0)
    nodes_df['unique_out_degree'] = tx_outgoing_grouped['to_address'].nunique().reindex(nodes_df.index).fillna(0)
    nodes_df['total_amount_outgoing'] = tx_outgoing_grouped['value'].sum().reindex(nodes_df.index).fillna(0)
    nodes_df['avg_amount_outgoing'] = tx_outgoing_grouped['value'].mean().reindex(nodes_df.index)
    nodes_df['max_amount_outgoing'] = tx_outgoing_grouped['value'].max().reindex(nodes_df.index)
    nodes_df['min_amount_outgoing'] = tx_outgoing_grouped['value'].min().reindex(nodes_df.index)
    nodes_df['avg_gas_price_sending'] = tx_outgoing_grouped['gas_price'].mean().reindex(nodes_df.index)
    nodes_df['total_gas_used_sending'] = tx_outgoing_grouped['receipt_gas_used'].sum().reindex(nodes_df.index)
    nodes_df['avg_gas_used_sending'] = tx_outgoing_grouped['receipt_gas_used'].mean().reindex(nodes_df.index)
    nodes_df['max_gas_used_sending'] = tx_outgoing_grouped['receipt_gas_used'].max().reindex(nodes_df.index)
    nodes_df['min_gas_used_sending'] = tx_outgoing_grouped['receipt_gas_used'].min().reindex(nodes_df.index)
    nodes_df['num_distinct_blocks_sent_from'] = tx_outgoing_grouped['block_number'].nunique().reindex(nodes_df.index)
    nodes_df['min_block_sent_from'] = tx_outgoing_grouped['block_number'].min().reindex(nodes_df.index)
    nodes_df['max_block_sent_from'] = tx_outgoing_grouped['block_number'].max().reindex(nodes_df.index)

    # --- incoming transaction features ---
    tx_incoming_grouped = df.groupby('to_address')
    nodes_df['in_degree'] = tx_incoming_grouped.size().reindex(nodes_df.index).fillna(0)
    nodes_df['unique_in_degree'] = tx_incoming_grouped['from_address'].nunique().reindex(nodes_df.index).fillna(0)
    nodes_df['total_amount_incoming'] = tx_incoming_grouped['value'].sum().reindex(nodes_df.index).fillna(0)
    nodes_df['avg_amount_incoming'] = tx_incoming_grouped['value'].mean().reindex(nodes_df.index)
    nodes_df['max_amount_incoming'] = tx_incoming_grouped['value'].max().reindex(nodes_df.index)
    nodes_df['min_amount_incoming'] = tx_incoming_grouped['value'].min().reindex(nodes_df.index)
    nodes_df['total_gas_used_on_received_txs'] = tx_incoming_grouped['receipt_gas_used'].sum().reindex(nodes_df.index)
    nodes_df['avg_gas_used_on_received_txs'] = tx_incoming_grouped['receipt_gas_used'].mean().reindex(nodes_df.index)
    nodes_df['max_gas_used_on_received_txs'] = tx_incoming_grouped['receipt_gas_used'].max().reindex(nodes_df.index)
    nodes_df['min_gas_used_on_received_txs'] = tx_incoming_grouped['receipt_gas_used'].min().reindex(nodes_df.index)
    nodes_df['num_distinct_blocks_received_in'] = tx_incoming_grouped['block_number'].nunique().reindex(nodes_df.index)
    nodes_df['min_block_received_in'] = tx_incoming_grouped['block_number'].min().reindex(nodes_df.index)
    nodes_df['max_block_received_in'] = tx_incoming_grouped['block_number'].max().reindex(nodes_df.index)

    # --- combined features ---
    nodes_df['all_degree'] = nodes_df['in_degree'] + nodes_df['out_degree']
    nodes_df['ether_balance'] = nodes_df['total_amount_incoming'] - nodes_df['total_amount_outgoing']
    nodes_df['total_amount'] = nodes_df['total_amount_incoming'] + nodes_df['total_amount_outgoing']

    nodes_df['block_span_outgoing'] = (nodes_df['max_block_sent_from'] - nodes_df['min_block_sent_from']).abs()
    nodes_df['block_span_incoming'] = (nodes_df['max_block_received_in'] - nodes_df['min_block_received_in']).abs()

    from_interactions = df[['from_address', 'block_number']].rename(columns={'from_address': 'address'})
    to_interactions = df[['to_address', 'block_number']].rename(columns={'to_address': 'address'})
    all_interactions_df = pd.concat([from_interactions, to_interactions]).dropna(subset=['address', 'block_number'])
    all_interactions_df['block_number'] = pd.to_numeric(all_interactions_df['block_number'], errors='coerce')
    all_interactions_df.dropna(subset=['block_number'], inplace=True) 

    if not all_interactions_df.empty:
        overall_block_stats = all_interactions_df.groupby('address')['block_number'].agg(
            num_distinct_blocks_overall='nunique',
            min_block_overall='min',
            max_block_overall='max'
        ).reindex(nodes_df.index)
        nodes_df = nodes_df.join(overall_block_stats)
    else:
        nodes_df['num_distinct_blocks_overall'] = 0
        nodes_df['min_block_overall'] = 0
        nodes_df['max_block_overall'] = 0
    
    nodes_df['num_distinct_blocks_overall'] = nodes_df['num_distinct_blocks_overall'].fillna(0)
    nodes_df['min_block_overall'] = nodes_df['min_block_overall'].fillna(0)
    nodes_df['max_block_overall'] = nodes_df['max_block_overall'].fillna(0)
    nodes_df['block_span_overall'] = (nodes_df['max_block_overall'] - nodes_df['min_block_overall']).abs()
    
    # ratio features
    nodes_df['ratio_sent_received_count'] = (nodes_df['out_degree'] / nodes_df['in_degree']).replace([np.inf, -np.inf], np.nan)
    nodes_df['ratio_sent_received_value'] = (nodes_df['total_amount_outgoing'] / nodes_df['total_amount_incoming']).replace([np.inf, -np.inf], np.nan)

    # --- proportion of unique counterparties ---
    nodes_df['prop_unique_out_degree'] = (nodes_df['unique_out_degree'] / nodes_df['out_degree']).replace([np.inf, -np.inf], np.nan)
    nodes_df['prop_unique_in_degree'] = (nodes_df['unique_in_degree'] / nodes_df['in_degree']).replace([np.inf, -np.inf], np.nan)

    # --- value concentration features ---
    nodes_df['outgoing_value_concentration'] = (nodes_df['max_amount_outgoing'] / nodes_df['total_amount_outgoing']).replace([np.inf, -np.inf], np.nan)
    nodes_df['incoming_value_concentration'] = (nodes_df['max_amount_incoming'] / nodes_df['total_amount_incoming']).replace([np.inf, -np.inf], np.nan)

    # --- self transaction features ---
    self_tx_df = df[df['from_address'] == df['to_address']]
    if not self_tx_df.empty:
        self_tx_grouped = self_tx_df.groupby('from_address')
        nodes_df['num_self_transactions'] = self_tx_grouped.size().reindex(nodes_df.index)
        nodes_df['total_value_self_transactions'] = self_tx_grouped['value'].sum().reindex(nodes_df.index)
    else:
        nodes_df['num_self_transactions'] = 0
        nodes_df['total_value_self_transactions'] = 0

    # --- network features ---
    edgelist_df = df[['from_address', 'to_address']].dropna().drop_duplicates()
    if not edgelist_df.empty:
        G_directed = nx.from_pandas_edgelist(edgelist_df, 'from_address', 'to_address', create_using=nx.DiGraph())
        G_undirected = G_directed.to_undirected()
        clustering_coeffs = nx.clustering(G_undirected)
        nodes_df['clustering_coefficient'] = nodes_df.index.map(clustering_coeffs)
        try:
            pagerank_coeffs = nx.pagerank(G_directed, alpha=0.85)
            nodes_df['pagerank'] = nodes_df.index.map(pagerank_coeffs)
        except nx.PowerIterationFailedConvergence:
            print("PageRank did not converge, dropping feature.")
        
    # --- scam features ---
    scam_category_map = {}
    if 'from_scam' in df.columns and 'from_category' in df.columns:
        from_scam_df = df[df['from_scam'] == 1][['from_address', 'from_category']]
        from_groups = from_scam_df.groupby('from_address')
        for addr, group in from_groups:
            first_cat = group['from_category'].dropna().values
            if len(first_cat) > 0:
                scam_category_map[addr] = first_cat[0]
    if 'to_scam' in df.columns and 'to_category' in df.columns:
        to_scam_df = df[df['to_scam'] == 1][['to_address', 'to_category']]
        to_groups = to_scam_df.groupby('to_address')
        for addr, group in to_groups:
            first_cat = group['to_category'].dropna().values
            if len(first_cat) > 0:
                scam_category_map[addr] = first_cat[0] 
    nodes_df['scam_category'] = nodes_df.index.map(scam_category_map)
    nodes_df['scam'] = nodes_df['scam_category'].notna().astype(int)

    # --- finalize dataFrame ---
    nodes_df.reset_index(inplace=True)

    # map address to node_id for edge list
    address_to_id = dict(zip(nodes_df['address'], nodes_df['node_id']))
    edgelist_df['fromId'] = edgelist_df['from_address'].map(address_to_id)
    edgelist_df['toId'] = edgelist_df['to_address'].map(address_to_id)
    edgelist_df = edgelist_df.dropna(subset=['fromId', 'toId']).astype({'fromId': int, 'toId': int})

    feature_columns_ordered = [
        'node_id', 'address', 'in_degree', 'out_degree', 'all_degree',
        'unique_in_degree', 'unique_out_degree',
        'prop_unique_in_degree', 'prop_unique_out_degree',
        'total_amount_incoming', 'total_amount_outgoing',
        'avg_amount_incoming', 'avg_amount_outgoing',
        'max_amount_incoming', 'min_amount_incoming',
        'max_amount_outgoing', 'min_amount_outgoing', 
        'incoming_value_concentration', 'outgoing_value_concentration',
        'total_amount', 'ether_balance',
        'ratio_sent_received_count', 'ratio_sent_received_value',
        'avg_gas_price_sending', 'total_gas_used_sending', 'avg_gas_used_sending',
        'max_gas_used_sending', 'min_gas_used_sending',
        'total_gas_used_on_received_txs', 'avg_gas_used_on_received_txs',
        'max_gas_used_on_received_txs', 'min_gas_used_on_received_txs',
        'num_distinct_blocks_sent_from', 'min_block_sent_from', 'max_block_sent_from', 'block_span_outgoing',
        'num_distinct_blocks_received_in', 'min_block_received_in', 'max_block_received_in', 'block_span_incoming',
        'num_distinct_blocks_overall', 'min_block_overall', 'max_block_overall', 'block_span_overall',
        'num_self_transactions', 'total_value_self_transactions',
        'clustering_coefficient', 'pagerank',
        'scam', 'scam_category'
    ]
    median_cols = [
        'ratio_sent_received_count', 'ratio_sent_received_value',
        'prop_unique_out_degree', 'prop_unique_in_degree',
        'outgoing_value_concentration', 'incoming_value_concentration',
        'avg_amount_outgoing', 'max_amount_outgoing', 'min_amount_outgoing',
        'avg_amount_incoming', 'max_amount_incoming', 'min_amount_incoming',
        'avg_gas_price_sending', 'avg_gas_used_sending', 
        'max_gas_used_sending', 'min_gas_used_sending',
        'avg_gas_used_on_received_txs',
        'max_gas_used_on_received_txs', 'min_gas_used_on_received_txs',
        'clustering_coefficient', 'pagerank'
    ]
    for col in median_cols:
        if col in nodes_df.columns:
            if pd.api.types.is_numeric_dtype(nodes_df[col]):
                medians = nodes_df.groupby('scam_category', dropna=False)[col].transform('median')
                nodes_df[col] = nodes_df[col].fillna(medians)
                
    for col in feature_columns_ordered:
        if col not in nodes_df.columns:
            if 'timestamp' in col or 'date' in col:
                nodes_df[col] = pd.NaT
            elif any(kw in col for kw in ['avg_', 'min_', 'max_', 'mean_', 'ratio_', 'balance', 'duration', 'interval', 'coefficient', 'concentration', 'pagerank', 'prop_']):
                nodes_df[col] = np.nan
            else:
                nodes_df[col] = 0

    for col in nodes_df.select_dtypes(include=np.number).columns:
        if col not in ['node_id', 'scam']:
            nodes_df[col] = nodes_df[col].fillna(0)
            
    if 'scam_category' in nodes_df.columns:
        nodes_df['scam_category'] = nodes_df['scam_category'].astype(object).fillna(pd.NA)
        
    # scale numeric features
    numeric_cols = [col for col in nodes_df.columns if pd.api.types.is_numeric_dtype(nodes_df[col]) and col not in ['node_id', 'address', 'scam']]
    scaler = StandardScaler()
    nodes_df[numeric_cols] = scaler.fit_transform(nodes_df[numeric_cols])

    final_cols = [col for col in feature_columns_ordered if col in nodes_df.columns]
    nodes_features_df = nodes_df[final_cols]

    return nodes_features_df, edgelist_df

def data_to_pyg(df_class_feature, df_edges, df_edge_attr=None, val_ratio=0.10, test_ratio=0.2):
    """Converts DataFrame of node features and edges to a PyTorch Geometric Data object.
    """
    edge_index = torch.tensor(
        np.stack([df_edges["fromId"].values, df_edges["toId"].values]), dtype=torch.long)
    
    feature_data = df_class_feature.drop(columns=['node_id', 'address', 'scam', 'scam_category'], errors='ignore')
    x = torch.tensor(feature_data.values, dtype=torch.float)
    y = torch.tensor(df_class_feature["scam"].values, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=df_edge_attr, y=y)
    
    if data.num_nodes * (1 - val_ratio - test_ratio) > 1:
        data = RandomNodeSplit(num_val=val_ratio, num_test=test_ratio)(data)
    else:
        raise ValueError("Not enough nodes for RandomNodeSplit with current ratios.")
    
    return data
