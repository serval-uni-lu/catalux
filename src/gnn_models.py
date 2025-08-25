import numpy as np
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Dropout, BatchNorm1d, ModuleList
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, SAGEConv, ChebConv

from .config import Config


class GCN(Module):
    def __init__(self, num_features: int, hidden_units: int, num_classes: int, 
                 dropout: float = 0.2, num_layers: int = 2, use_batch_norm: bool = False):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        
        # build layers
        self.convs = ModuleList()
        self.batch_norms = ModuleList() if use_batch_norm else None
        
        # first layer
        self.convs.append(GCNConv(num_features, hidden_units, cached=True))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_units))
        
        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_units, hidden_units, cached=True))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_units))
        
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x_prev = x if i > 0 else None
            
            x = conv(x, edge_index)
            x = F.relu(x)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.dropout(x)

            # residual connection
            if x_prev is not None and x_prev.shape == x.shape:
                x = x + x_prev
        
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class GAT(Module):
    def __init__(self, num_features: int, edge_dim: int, hidden_units: int, num_classes: int,
                 heads: int = 4, gat_dropout: float = 0.2, dropout: float = 0.2, 
                 num_layers: int = 2, use_batch_norm: bool = False):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.hidden_units = hidden_units
        
        self.convs = ModuleList()
        self.batch_norms = ModuleList() if use_batch_norm else None
        
        # first layer
        self.convs.append(GATConv(num_features, hidden_units // heads, heads=heads,
                                 dropout=gat_dropout, edge_dim=edge_dim, concat=True))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_units))
        
        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_units, hidden_units // heads, heads=heads,
                                     dropout=gat_dropout, edge_dim=edge_dim, concat=True))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_units))
        
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x_prev = x if i > 0 and x.shape[-1] == self.hidden_units else None
            
            x = F.elu(conv(x, edge_index, edge_attr))
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.dropout(x)
            
            # residual connection
            if x_prev is not None and x_prev.shape == x.shape:
                x = x + x_prev
        
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class GATv2(Module):
    def __init__(self, num_features: int, edge_dim: int, hidden_units: int, num_classes: int,
                 heads: int = 4, gat_dropout: float = 0.2, dropout: float = 0.2,
                 num_layers: int = 2, use_batch_norm: bool = False):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.hidden_units = hidden_units
        
        self.convs = ModuleList()
        self.batch_norms = ModuleList() if use_batch_norm else None
        
        # first layer
        self.convs.append(GATv2Conv(num_features, hidden_units // heads, heads=heads,
                                   dropout=gat_dropout, edge_dim=edge_dim, concat=True))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_units))
        
        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_units, hidden_units // heads, heads=heads,
                                       dropout=gat_dropout, edge_dim=edge_dim, concat=True))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_units))
        
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x_prev = x if i > 0 and x.shape[-1] == self.hidden_units else None
            
            x = F.elu(conv(x, edge_index, edge_attr))
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.dropout(x)
            
            # residual connection
            if x_prev is not None and x_prev.shape == x.shape:
                x = x + x_prev
        
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class SAGE(Module):
    def __init__(self, num_features: int, hidden_units: int, num_classes: int,
                 dropout: float = 0.2, num_layers: int = 2, use_batch_norm: bool = True):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        
        self.convs = ModuleList()
        self.batch_norms = ModuleList() if use_batch_norm else None
        
        # first layer
        self.convs.append(SAGEConv(num_features, hidden_units, normalize=True))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_units))
        
        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_units, hidden_units, normalize=True))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_units))
        
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_residual = None
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = self.dropout(x)
            
            # residual connection
            if i == 0:
                x_residual = x
            elif i == self.num_layers - 1 and x_residual is not None:
                x = x + x_residual
        
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class Chebyshev(Module):
    def __init__(self, num_features: int, hidden_units: int, num_classes: int,
                 K: List[int] = [1, 2], dropout: float = 0.2, num_layers: int = 2, 
                 use_batch_norm: bool = False):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        
        self.convs = ModuleList()
        self.batch_norms = ModuleList() if use_batch_norm else None
        
        # ensure we have enough k values for layers
        if len(K) < num_layers:
            K = K + [K[-1]] * (num_layers - len(K))
        
        # first layer
        self.convs.append(ChebConv(num_features, hidden_units, K=K[0]))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_units))
        
        # hidden layers
        for i in range(1, num_layers):
            self.convs.append(ChebConv(hidden_units, hidden_units, K=K[i]))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_units))
        
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x_prev = x if i > 0 else None
            
            x = F.relu(conv(x, edge_index))
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.dropout(x)
            
            # residual connection
            if x_prev is not None and x_prev.shape == x.shape:
                x = x + x_prev
        
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


def get_gnn_constructors(num_features: int, edge_dim: int, config: Config) -> Dict[str, callable]:
    """Get model constructors with additional parameters."""
    
    # get parameters with defaults
    num_layers = getattr(config, 'layers', 2)
    use_bn = getattr(config, 'use_batch_norm', False)
    
    return {
        'GCN': lambda: GCN(
            num_features, config.hidden_units, config.num_classes,
            dropout=config.dropout, num_layers=num_layers, use_batch_norm=use_bn
        ),
        'GAT': lambda: GAT(
            num_features, edge_dim, config.hidden_units, config.num_classes,
            heads=config.gat_heads, gat_dropout=config.gat_dropout,
            dropout=config.dropout, num_layers=num_layers, use_batch_norm=use_bn
        ),
        'GATv2': lambda: GATv2(
            num_features, edge_dim, config.hidden_units, config.num_classes,
            heads=config.gatv2_heads, gat_dropout=config.gatv2_dropout,
            dropout=config.dropout, num_layers=num_layers, use_batch_norm=use_bn
        ),
        'SAGE': lambda: SAGE(
            num_features, config.hidden_units, config.num_classes,
            dropout=config.dropout, num_layers=num_layers, use_batch_norm=True
        ), 
        'Chebyshev': lambda: Chebyshev(
            num_features, config.hidden_units, config.num_classes,
            K=config.cheb_k, dropout=config.dropout, num_layers=num_layers,
            use_batch_norm=use_bn
        )
    }
    

def load_pretrained(model_path: str, num_features: int, edge_dim: int, config: Config) -> Module:
    """Load a pretrained model from the specified path."""
    model_constructors = get_gnn_constructors(num_features, edge_dim, config)
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
    graph = graph.to(next(model.parameters()).device)
    
    with torch.no_grad():
        x, edge_index = graph.x, graph.edge_index
        
        if hasattr(model, 'convs') and hasattr(model.convs[0], 'edge_dim') and graph.edge_attr is not None:
            logits = model(x, edge_index, graph.edge_attr)
        else:
            logits = model(x, edge_index)
        
        probs = F.softmax(logits[ids], dim=-1)[:, 1]
        return probs.cpu().numpy()