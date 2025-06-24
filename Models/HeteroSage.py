import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, SAGEConv

class HeteroSAGE(nn.Module):
    def __init__(
        self, 
        metadata,                # (node_types, edge_types)
        out_dim,
        hidden_dim, 
        x_dict,
        target_feat="author", 
        num_layers=2, 
        dropout=0.3, 
        with_non_linear=True
    ):
        super().__init__()
        self.target_feat = target_feat
        self.node_types, self.edge_types = metadata
        self.dropout = dropout

        # Initial linear projection for each node type
        self.embeddings = nn.ModuleDict({
            ntype: nn.Linear(x_dict[ntype].shape[1], hidden_dim)
            for ntype in self.node_types
        })

        # Stack of hetero SAGE layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroGraphConv({
                etype: SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean')
                for etype in self.edge_types
            }, aggregate='sum')
            self.convs.append(conv)

        self.layer_norms = nn.ModuleDict({
            ntype: nn.LayerNorm(hidden_dim)
            for ntype in self.node_types
        })

        # Output projection per node type
        self.output_projs = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU() if with_non_linear else nn.Identity(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim if ntype == target_feat else hidden_dim)
            )
            for ntype in self.node_types
        })

    def forward(self, graph, x_dict):
        # Project input features to hidden space
        x_dict = {
            ntype: self.embeddings[ntype](x)
            for ntype, x in x_dict.items()
        }

        # Multi-layer SAGE propagation
        for conv in self.convs:
            x_prev = {k: v.clone() for k, v in x_dict.items()}
            x_dict = conv(graph, x_dict)

            # Residual connection + LayerNorm + ReLU
            x_dict = {
                ntype: self.layer_norms[ntype](F.relu(h) + x_prev[ntype])
                for ntype, h in x_dict.items()
            }

            # Dropout
            x_dict = {
                ntype: F.dropout(h, p=self.dropout, training=self.training)
                for ntype, h in x_dict.items()
            }

        # Final projections
        out_dict = {
            ntype: self.output_projs[ntype](h)
            for ntype, h in x_dict.items()
        }

        # Log-softmax on target node type
        if self.target_feat in out_dict:
            out_dict[self.target_feat] = F.log_softmax(out_dict[self.target_feat], dim=1)

        return out_dict
