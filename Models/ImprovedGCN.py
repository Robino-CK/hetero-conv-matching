import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, SAGEConv

class ImprovedGCN(nn.Module):
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
        hidden_channels = hidden_dim
        self.target_feat = target_feat
        num_classes = out_dim
        self.node_types, self.edge_types = metadata
        
        self.node_types, self.edge_types = metadata
        self.dropout = dropout

        # Get input dimensions from x_dict
        self.embeddings = nn.ModuleDict({
            ntype: nn.Linear(x_dict[ntype].shape[1], hidden_channels)
            for ntype in self.node_types
        })

        # Stack of hetero SAGE layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_layer = HeteroGraphConv({
                etype: SAGEConv(hidden_channels, hidden_channels, aggregator_type='mean')
                for etype in self.edge_types
            }, aggregate='sum')
            self.convs.append(conv_layer)

        self.layer_norms = nn.ModuleDict({
            ntype: nn.LayerNorm(hidden_channels)
            for ntype in self.node_types
        })

        # Output projections (with or without non-linearity)
        self.output_projs = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU() if with_non_linear else nn.Identity(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_classes if ntype == target_feat else hidden_channels)
            )
            for ntype in self.node_types
        })

    def forward(self, graph, x_dict):
        # Initial linear projection for each node type
        x_dict = {
            ntype: self.embeddings[ntype](x)
            for ntype, x in x_dict.items()
        }

        for conv in self.convs:
            x_prev = {k: v.clone() for k, v in x_dict.items()}
            x_dict = conv(graph, x_dict)  # heterogeneous graph convolution

            # Residual + LayerNorm + ReLU
            x_dict = {
                ntype: self.layer_norms[ntype](F.relu(h) + x_prev[ntype])
                for ntype, h in x_dict.items()
            }

            # Dropout
            x_dict = {
                ntype: F.dropout(h, p=self.dropout, training=self.training)
                for ntype, h in x_dict.items()
            }

        # Output projection
        out_dict = {
            ntype: self.output_projs[ntype](h)
            for ntype, h in x_dict.items()
        }

        # Apply log_softmax only for classification target
        if self.target_feat in out_dict:
            out_dict[self.target_feat] = F.log_softmax(out_dict[self.target_feat], dim=1)

        return out_dict
