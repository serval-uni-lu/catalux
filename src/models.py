import numpy as np
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Dropout, BatchNorm1d
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, SAGEConv, ChebConv

from .config import Config

class GCNConvolution(Module):
    def __init__(self, num_features: int, hidden_units: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_units, cached=True)
        self.conv2 = GCNConv(hidden_units, hidden_units, cached=True)
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x_residual = x
        
        x = self.conv2(x, edge_index)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + x_residual
        
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

class GATConvolution(Module):
    def __init__(self, num_features: int, edge_dim: int, hidden_units: int, num_classes: int,
                 heads: int = 4, gat_dropout: float = 0.2, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_units // heads, heads=heads, 
                            dropout=gat_dropout, edge_dim=edge_dim, concat=True)
        self.conv2 = GATConv(hidden_units, hidden_units // heads, heads=heads,
                            dropout=gat_dropout, edge_dim=edge_dim, concat=True)
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x_residual = x

        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = x + x_residual
        
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

class GATv2Convolution(Module):
    def __init__(self, num_features: int, edge_dim: int, hidden_units: int, num_classes: int,
                 heads: int = 4, gat_dropout: float = 0.2, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, hidden_units // heads, heads=heads,
                              dropout=gat_dropout, edge_dim=edge_dim, concat=True)
        self.conv2 = GATv2Conv(hidden_units, hidden_units // heads, heads=heads,
                              dropout=gat_dropout, edge_dim=edge_dim, concat=True)
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x_residual = x

        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = x + x_residual
        
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

class SAGEConvolution(Module):
    def __init__(self, num_features: int, hidden_units: int, num_classes: int,
                 dropout: float = 0.2, use_batch_norm: bool = False):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_units, normalize=True)
        self.conv2 = SAGEConv(hidden_units, hidden_units, normalize=True)
        self.use_batch_norm = use_batch_norm
        
        if use_batch_norm:
            self.bn1 = BatchNorm1d(hidden_units)
            self.bn2 = BatchNorm1d(hidden_units)
        
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x_residual = x
        
        x = self.conv2(x, edge_index)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + x_residual
        
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

class ChebyshevConvolution(Module):
    def __init__(self, num_features: int, hidden_units: int, num_classes: int,
                 K: List[int] = [1, 2], dropout: float = 0.2):
        super().__init__()
        self.conv1 = ChebConv(num_features, hidden_units, K=K[0])
        self.conv2 = ChebConv(hidden_units, hidden_units, K=K[1])
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x_residual = x
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = x + x_residual
        
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

def get_model_constructors(num_features: int, edge_dim: int, config: Config) -> Dict[str, callable]:
    """Get model constructors for different architectures."""
    return {
        'GCN': lambda: GCNConvolution(
            num_features, config.hidden_units, config.num_classes,
            dropout=config.dropout
        ),
        'GAT': lambda: GATConvolution(
            num_features, edge_dim, config.hidden_units, config.num_classes,
            heads=config.gat_heads, gat_dropout=config.gat_dropout,
            dropout=config.dropout
        ),
        'GATv2': lambda: GATv2Convolution(
            num_features, edge_dim, config.hidden_units, config.num_classes,
            heads=config.gatv2_heads, gat_dropout=config.gatv2_dropout,
            dropout=config.dropout
        ),
        'SAGE': lambda: SAGEConvolution(
            num_features, config.hidden_units, config.num_classes,
            dropout=config.dropout
        ), 
        'Chebyshev': lambda: ChebyshevConvolution(
            num_features, config.hidden_units, config.num_classes,
            K=config.cheb_k, dropout=config.dropout
        )
    }

def load_pretrained(model_path: str, num_features: int, edge_dim: int, config: Config) -> Module:
    """Load a pretrained model from the specified path.
    """
    model_constructors = get_model_constructors(num_features, edge_dim, config)
    model_name = model_path.split('/')[-1].split('.')[0]
    if model_name not in model_constructors:
        raise ValueError(f"Model {model_name} not found in available constructors.")
    
    device = config.get_device()
    model = model_constructors[model_name]()
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded model: {model_name} from {model_path} on device: {device}")
    
    return model

def predict_for_ids(model, graph, ids: list[int]) -> np.ndarray:
    """Get predictions for specific node IDs."""
    model.eval()
    device = next(model.parameters()).device
    graph = graph.to(device)
    
    with torch.no_grad():
        x = graph.x
        edge_index = graph.edge_index
        
        if hasattr(model.conv1, 'edge_dim') and graph.edge_attr is not None:
            logits = model(x, edge_index, graph.edge_attr)
        else:
            logits = model(x, edge_index)
        
        probs = F.softmax(logits, dim=-1)[:, 1]  # probability of scam class
        return probs[ids].cpu().numpy()