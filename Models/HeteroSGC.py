import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, GraphConv

class HeteroSGC(nn.Module):
    def __init__(
        self, 
        metadata,                # (node_types, edge_types)
        out_dim,
        hidden_dim, 
        x_dict,
        target_feat="author", 
        num_layers=2,            # k-hop SGC
        dropout=0.3, 
        with_non_linear=True
    ):
        super().__init__()
        self.target_feat = target_feat
        self.node_types, self.edge_types = metadata
        self.dropout = dropout
        self.num_layers = num_layers

        # Initial feature projection (to hidden_dim)
        self.embeddings = nn.ModuleDict({
            ntype: nn.Linear(x_dict[ntype].shape[1], hidden_dim)
            for ntype in self.node_types
        })

        # A single shared GraphConv layer for SGC (no activation)
        self.sgcs = nn.ModuleList([
            HeteroGraphConv({
                etype: GraphConv(hidden_dim, hidden_dim, norm='both', weight=False, bias=False)
                for etype in self.edge_types
            }, aggregate='sum')
            for _ in range(num_layers)
        ])

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
        # Initial projection
        x_dict = {
            ntype: self.embeddings[ntype](x)
            for ntype, x in x_dict.items()
        }

        # K-hop propagation (no activation)
        for conv in self.sgcs:
            x_dict = conv(graph, x_dict)

        # Optional normalization after aggregation
        x_dict = {
            ntype: self.layer_norms[ntype](x)
            for ntype, x in x_dict.items()
        }

        # Dropout before output projection
        x_dict = {
            ntype: F.dropout(x, p=self.dropout, training=self.training)
            for ntype, x in x_dict.items()
        }

        # Output projection
        out_dict = {
            ntype: self.output_projs[ntype](x)
            for ntype, x in x_dict.items()
        }

        # Log softmax for classification
        if self.target_feat in out_dict:
            out_dict[self.target_feat] = F.log_softmax(out_dict[self.target_feat], dim=1)

        return out_dict
