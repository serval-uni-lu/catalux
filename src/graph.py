import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Callable, Dict, Any


class Graph:
    """Graph structure manager with gradient tracking for PyTorch Geometric."""
    
    def __init__(self, data, requires_grad: bool = True):
        """
        Initialize graph from PyTorch Geometric data object.
        
        Args:
            data: PyTorch Geometric Data object
            requires_grad: Whether to track gradients for node/edge features
        """
        if hasattr(data, 'clone'):
            self.data = data.clone()
        else:
            import torch_geometric.data
            self.data = torch_geometric.data.Data()
            
            # copy tensor attributes
            for key in data.keys:
                value = data[key]
                if isinstance(value, torch.Tensor):
                    self.data[key] = value.clone()
                else:
                    self.data[key] = value
        
        self._requires_grad = requires_grad
        self._num_nodes = self.data.x.shape[0] if self.data.x is not None else 0

        if self._requires_grad:
            self._setup_gradients()
    
    def _setup_gradients(self):
        """Configure gradient tracking for features."""
        if self.data.x is not None:
            self.data.x = self.data.x.detach().requires_grad_(True)
        if hasattr(self.data, 'edge_attr') and self.data.edge_attr is not None:
            self.data.edge_attr = self.data.edge_attr.detach().requires_grad_(True)
    
    # ============= properties =============
    
    @property
    def x(self) -> Optional[torch.Tensor]:
        """Node features."""
        return self.data.x
    
    @x.setter
    def x(self, value: torch.Tensor):
        """Set node features with gradient tracking."""
        self.data.x = value
        if self._requires_grad and value is not None:
            self.data.x = self.data.x.detach().requires_grad_(True)
    
    @property
    def edge_index(self) -> torch.Tensor:
        """Edge connectivity."""
        return self.data.edge_index
    
    @edge_index.setter
    def edge_index(self, value: torch.Tensor):
        """Set edge connectivity."""
        self.data.edge_index = value
    
    @property
    def edge_attr(self) -> Optional[torch.Tensor]:
        """Edge features."""
        return getattr(self.data, 'edge_attr', None)
    
    @edge_attr.setter
    def edge_attr(self, value: Optional[torch.Tensor]):
        """Set edge features with gradient tracking."""
        self.data.edge_attr = value
        if self._requires_grad and value is not None:
            self.data.edge_attr = self.data.edge_attr.detach().requires_grad_(True)
            
    @property
    def y(self) -> Optional[torch.Tensor]:
        """Node labels."""
        return getattr(self.data, 'y', None)

    @y.setter
    def y(self, value: Optional[torch.Tensor]):
        """Set node labels with gradient tracking."""
        self.data.y = value
        if self._requires_grad and value is not None:
            self.data.y = self.data.y.detach().requires_grad_(True)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._num_nodes
    
    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self.edge_index.shape[1]
    
    @property
    def device(self) -> torch.device:
        """Device where the graph is stored."""
        return self.x.device if self.x is not None else torch.device('cpu')
    
    @property
    def requires_grad(self) -> bool:
        """Whether gradient tracking is enabled."""
        return self._requires_grad
    
    # ============= node operations =============
    
    def add_node(self, features: Optional[torch.Tensor] = None) -> int:
        """
        Add a single node to the graph with optimized memory usage.
        
        Args:
            features: Node features (optional). If None, uses zeros.
        
        Returns:
            Index of the newly added node
        """
        if features is None:
            features = torch.zeros(1, self.x.shape[1], dtype=self.x.dtype, device=self.device)
        elif features.dim() == 1:
            features = features.unsqueeze(0)
        
        features = features.to(self.device)
        
        self.x = torch.cat([self.x, features], dim=0)
        self._num_nodes += 1

        return int(self._num_nodes - 1)

    def add_nodes(self, features: torch.Tensor) -> List[int]:
        """
        Add multiple nodes efficiently.
        
        Args:
            features: Node features tensor (num_nodes, feature_dim)
        
        Returns:
            List of indices for newly added nodes
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        start_id = self._num_nodes
        num_new = features.shape[0]
        
        self.x = torch.cat([self.x, features], dim=0)
        self._num_nodes += num_new
        
        return list(range(start_id, self._num_nodes))
    
    def update_node_features(self, node_ids: Union[int, List[int]], features: torch.Tensor):
        """
        Update features for specified nodes.
        
        Args:
            node_ids: Single node index or list of indices
            features: New features for the nodes
        """
        if isinstance(node_ids, int):
            node_ids = [node_ids]
            if features.dim() == 1:
                features = features.unsqueeze(0)
        
        if any(idx >= self._num_nodes or idx < 0 for idx in node_ids):
            raise ValueError(f"Invalid node indices. Graph has {self._num_nodes} nodes.")
        
        if self._requires_grad:
            # clone to maintain gradient graph
            new_x = self.x.clone()
            new_x[node_ids] = features
            self.x = new_x
        else:
            # direct assignment when gradients not needed
            self.x[node_ids] = features
    
    def get_node_features(self, node_ids: Optional[Union[int, List[int]]] = None) -> torch.Tensor:
        """
        Get features for specified nodes.
        
        Args:
            node_ids: Node indices. If None, returns all node features.
        
        Returns:
            Node features tensor
        """
        if node_ids is None:
            return self.x
        if isinstance(node_ids, int):
            return self.x[node_ids:node_ids+1]
        return self.x[node_ids]
    
    # ============= edge operations =============
    
    def add_edge(self, src: int, dst: int, features: Optional[torch.Tensor] = None):
        """
        Add a single edge to the graph with device consistency.
        
        Args:
            src: Source node index
            dst: Destination node index
            features: Edge features (optional)
        """
        new_edge = torch.tensor([[src], [dst]], dtype=self.edge_index.dtype, device=self.device)
        self.edge_index = torch.cat([self.edge_index, new_edge], dim=1)
        
        if features is not None:
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            features = features.to(self.device)
            
            if self.edge_attr is None:
                self.edge_attr = features
            else:
                self.edge_attr = torch.cat([self.edge_attr, features], dim=0)
    
    def add_edges(self, edge_index: torch.Tensor, features: Optional[torch.Tensor] = None):
        """
        Add multiple edges efficiently.
        
        Args:
            edge_index: Edge connectivity tensor (2, num_edges) or (num_edges, 2)
            features: Edge features (optional)
        """
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t()
        
        self.edge_index = torch.cat([self.edge_index, edge_index], dim=1)
        
        if features is not None:
            if self.edge_attr is None:
                self.edge_attr = features
            else:
                self.edge_attr = torch.cat([self.edge_attr, features], dim=0)
    
    def delete_edge(self, edge_idx: int):
        """
        Delete a specific edge by its index.
        
        Args:
            edge_idx: Index of the edge to remove
        """
        if edge_idx < 0 or edge_idx >= self.num_edges:
            raise ValueError(f"Invalid edge index {edge_idx}. Graph has {self.num_edges} edges.")
        
        # create mask to exclude the specified edge
        edge_mask = torch.ones(self.num_edges, dtype=torch.bool, device=self.device)
        edge_mask[edge_idx] = False
        
        # remove edge from edge_index
        self.edge_index = self.edge_index[:, edge_mask]
        
        # remove corresponding edge attributes if they exist
        if self.edge_attr is not None:
            if self._requires_grad:
                # maintain gradient tracking
                new_edge_attr = self.edge_attr[edge_mask]
                self.edge_attr = new_edge_attr
            else:
                self.edge_attr = self.edge_attr[edge_mask]
    
    # ============= gradient operations =============
    
    def compute_gradients(
        self,
        model: torch.nn.Module,
        node_ids: Optional[Union[int, List[int]]] = None,
        target_class: Optional[Union[int, torch.Tensor]] = None,
        loss_fn: Optional[Callable] = None,
        return_probabilities: bool = True,
        retain_graph: bool = False,
        create_graph: bool = False,
        return_edge_grads: bool = False,
        **model_kwargs
    ) -> Dict[str, Any]:
        """
        Compute gradients for specified nodes.
        
        Args:
            model: GNN model for forward pass
            node_ids: Node indices (None for all nodes)
            target_class: Target class for loss computation
            loss_fn: Custom loss function (default: NLL loss)
            return_probabilities: Whether to return output probabilities
            retain_graph: Keep computation graph for multiple backward passes
            create_graph: Create graph of gradient computation
            return_edge_grads: Whether to return edge feature gradients
            **model_kwargs: Additional arguments for model forward pass
        
        Returns:
            Dictionary containing:
                - 'gradients': Node feature gradients
                - 'probabilities': Output probabilities (optional)
                - 'logits': Raw model outputs
                - 'loss': Loss value
                - 'edge_gradients': Edge feature gradients (optional)
        """
        if not self._requires_grad:
            raise RuntimeError("Graph was initialized with requires_grad=False")
        
        if node_ids is None:
            node_ids = list(range(self._num_nodes))
        elif isinstance(node_ids, int):
            node_ids = [node_ids]
        
        if any(idx >= self._num_nodes or idx < 0 for idx in node_ids):
            raise ValueError(f"Invalid node indices. Graph has {self._num_nodes} nodes.")
        
        self._setup_gradients()
        
        model.eval()
        model.zero_grad()
        
        if self.edge_attr is not None and self._model_accepts_edge_attr(model):
            logits = model(self.x, self.edge_index, self.edge_attr, **model_kwargs)
        else:
            logits = model(self.x, self.edge_index, **model_kwargs)
        
        if target_class is None:
            target = torch.zeros(len(node_ids), dtype=torch.long, device=self.device)
        elif isinstance(target_class, int):
            target = torch.full((len(node_ids),), target_class, dtype=torch.long, device=self.device)
        else:
            target = target.to(self.device)
        
        if loss_fn is None:
            loss = F.nll_loss(F.log_softmax(logits[node_ids], dim=-1), target)
        else:
            loss = loss_fn(logits[node_ids], target)
        
        loss.backward(retain_graph=retain_graph, create_graph=create_graph)
        
        result = {
            'gradients': self.x.grad[node_ids].clone() if self.x.grad is not None else None,
            'logits': logits[node_ids].detach(),
            'loss': loss.item()
        }
        
        if return_probabilities:
            result['probabilities'] = torch.softmax(logits[node_ids], dim=-1).detach()
        
        if return_edge_grads and self.edge_attr is not None and self.edge_attr.grad is not None:
            result['edge_gradients'] = self.edge_attr.grad.clone()
        
        return result
    
    def _model_accepts_edge_attr(self, model: torch.nn.Module) -> bool:
        """Check if model accepts edge attributes."""
        import inspect
        sig = inspect.signature(model.forward)
        return 'edge_attr' in sig.parameters or 'edge_weight' in sig.parameters
    
    # ============= utility methods =============
    
    def validate(self) -> bool:
        """Validate graph consistency."""
        if self.x is None:
            return False
        
        # check node count consistency
        if self.x.shape[0] != self._num_nodes:
            return False
        
        # check edge indices are valid
        if self.edge_index.max() >= self._num_nodes:
            return False
        
        # check edge attributes consistency
        if self.edge_attr is not None and self.edge_attr.shape[0] != self.num_edges:
            return False
        
        # check gradient tracking
        if self._requires_grad:
            if not self.x.requires_grad:
                return False
            if self.edge_attr is not None and not self.edge_attr.requires_grad:
                return False
        
        return True
    
    def detach(self) -> 'Graph':
        """Create a detached copy without gradient tracking."""
        if hasattr(self.data, 'clone'):
            detached_data = self.data.clone()
            # detach all tensor attributes
            for key in detached_data.keys:
                if isinstance(detached_data[key], torch.Tensor):
                    detached_data[key] = detached_data[key].detach()
        else:
            import torch_geometric.data
            detached_data = torch_geometric.data.Data()

            # copy tensor attributes
            for key in self.data.keys:
                value = self.data[key]
                if isinstance(value, torch.Tensor):
                    detached_data[key] = value.detach()
                else:
                    detached_data[key] = value
        
        return Graph(detached_data, requires_grad=False)
    
    def clone(self) -> 'Graph':
        """Create a deep copy of the graph."""
        if hasattr(self.data, 'clone'):
            cloned_data = self.data.clone()
        else:
            import torch_geometric.data
            cloned_data = torch_geometric.data.Data()
            
            # copy tensor attributes
            for key in self.data.keys:
                value = self.data[key]
                if isinstance(value, torch.Tensor):
                    cloned_data[key] = value.clone()
                else:
                    cloned_data[key] = value
        
        return Graph(cloned_data, requires_grad=self._requires_grad)
    
    def to(self, device: Union[str, torch.device]) -> 'Graph':
        """Move graph to specified device with proper memory alignment."""
        if isinstance(device, str):
            device = torch.device(device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        self.data = self.data.to(device)
        if self._requires_grad:
            self._setup_gradients()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        return self
    
    def cpu(self) -> 'Graph':
        """Move graph to CPU."""
        return self.to('cpu')
    
    def cuda(self, device: Optional[int] = None) -> 'Graph':
        """Move graph to CUDA device."""
        if device is None:
            return self.to('cuda')
        return self.to(f'cuda:{device}')
    
    def memory_usage(self) -> dict:
        """Get memory usage statistics for the graph."""
        stats = {
            'nodes': self.x.numel() * self.x.element_size() if self.x is not None else 0,
            'edges': self.edge_index.numel() * self.edge_index.element_size(),
            'edge_attr': self.edge_attr.numel() * self.edge_attr.element_size() if self.edge_attr is not None else 0,
            'device': str(self.device)
        }
        stats['total'] = sum(v for k, v in stats.items() if isinstance(v, int))
        return stats
    
    def __repr__(self) -> str:
        """String representation of the graph."""
        return (f"Graph(num_nodes={self.num_nodes}, num_edges={self.num_edges}, "
                f"device={self.device}, requires_grad={self.requires_grad})")
    
    def __len__(self) -> int:
        """Number of nodes in the graph."""
        return self._num_nodes