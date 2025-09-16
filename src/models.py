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
        
        self.convs = ModuleList()
        self.batch_norms = ModuleList() if use_batch_norm else None
        
        self.convs.append(GCNConv(num_features, hidden_units))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_units))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_units, hidden_units))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_units))
        
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.dropout(x)
        
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class GAT(Module):
    def __init__(self, num_features: int, edge_dim: int, hidden_units: int, num_classes: int,
                 heads: int = 4, gat_dropout: float = 0.2, dropout: float = 0.2, 
                 num_layers: int = 2, use_batch_norm: bool = False):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        
        self.convs = ModuleList()
        self.batch_norms = ModuleList() if use_batch_norm else None
        
        self.convs.append(GATConv(num_features, hidden_units // heads, heads=heads,
                                 dropout=gat_dropout, edge_dim=edge_dim, concat=True))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_units))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_units, hidden_units // heads, heads=heads,
                                     dropout=gat_dropout, edge_dim=edge_dim, concat=True))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_units))
        
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index, edge_attr))
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.dropout(x)
        
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class GATv2(Module):
    def __init__(self, num_features: int, edge_dim: int, hidden_units: int, num_classes: int,
                 heads: int = 4, gat_dropout: float = 0.2, dropout: float = 0.2,
                 num_layers: int = 2, use_batch_norm: bool = False):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        
        self.convs = ModuleList()
        self.batch_norms = ModuleList() if use_batch_norm else None
        
        self.convs.append(GATv2Conv(num_features, hidden_units // heads, heads=heads,
                                   dropout=gat_dropout, edge_dim=edge_dim, concat=True))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_units))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_units, hidden_units // heads, heads=heads,
                                       dropout=gat_dropout, edge_dim=edge_dim, concat=True))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_units))
        
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index, edge_attr))
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.dropout(x)
        
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
        
        self.convs.append(SAGEConv(num_features, hidden_units))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_units))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_units, hidden_units))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_units))
        
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class Chebyshev(Module):
    def __init__(self, num_features: int, hidden_units: int, num_classes: int,
                 K: List[int] = [2, 3], dropout: float = 0.2, num_layers: int = 2, 
                 use_batch_norm: bool = False):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        
        self.convs = ModuleList()
        self.batch_norms = ModuleList() if use_batch_norm else None
        
        if len(K) < num_layers:
            K = K + [K[-1]] * (num_layers - len(K))
        
        self.convs.append(ChebConv(num_features, hidden_units, K=K[0]))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_units))
        
        for i in range(1, num_layers):
            self.convs.append(ChebConv(hidden_units, hidden_units, K=K[i]))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_units))
        
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_units, num_classes)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.dropout(x)
        
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


def get_gnn_constructors(num_features: int, edge_dim: int, config: Config) -> Dict[str, callable]:
    num_layers = config.layers
    use_bn = config.use_batch_norm
    
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


def load_model(model_name: str, num_features: int, edge_dim: int, config: Config) -> Module:
    from pathlib import Path
    
    model_path = Path(f"models/{model_name}/model.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    constructors = get_gnn_constructors(num_features, edge_dim, config)
    if model_name not in constructors:
        raise ValueError(f"Unknown model: {model_name}")
    
    device = config.get_device()
    model = constructors[model_name]()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model