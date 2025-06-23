
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, HeteroGraphConv


import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphConv
from dgl.nn.pytorch import  GraphConv

class HeteroGCNCiteerCrazy(nn.Module):
    def __init__(self, x_dict, hidden_dim, out_dim, metadata, target_feat, dropout=0.5):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        # Project input features for each node type to hidden_dim
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(x_dict[ntype].shape[1], hidden_dim)
            for ntype in x_dict
        })

        # GATConv for attention-based relation-specific message passing
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(hidden_dim, hidden_dim, heads=2, concat=False)
            for rel in self.edge_types
        }, aggregate='sum')

        self.norm1 = nn.ModuleDict({
            ntype: nn.LayerNorm(hidden_dim) for ntype in self.node_types
        })

        self.conv2 = HeteroGraphConv({
            rel: GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
            for rel in self.edge_types
        }, aggregate='sum')

        self.norm2 = nn.ModuleDict({
            ntype: nn.LayerNorm(hidden_dim) for ntype in self.node_types
        })

        # Type-specific output MLPs
        self.out_mlp = nn.ModuleDict({
            ntype: nn.Linear(hidden_dim, out_dim) for ntype in self.node_types
        })

    def forward(self, graph, x_dict):
        # Project inputs
        x_dict = {
            ntype: self.input_proj[ntype](x)
            for ntype, x in x_dict.items()
        }

        # conv1 + residual + norm + dropout
        h_dict1 = self.conv1(graph, x_dict)
        h_dict1 = {
            ntype: self.norm1[ntype](F.relu(h)) for ntype, h in h_dict1.items()
        }
        h_dict1 = {
            ntype: F.dropout(h, p=self.dropout, training=self.training)
            for ntype, h in h_dict1.items()
        }
        h_dict1 = {
            ntype: h + x_dict[ntype]  # residual connection
            for ntype, h in h_dict1.items()
        }

        # conv2 + residual + norm + dropout
        h_dict2 = self.conv2(graph, h_dict1)
        h_dict2 = {
            ntype: self.norm2[ntype](F.relu(h)) for ntype, h in h_dict2.items()
        }
        h_dict2 = {
            ntype: F.dropout(h, p=self.dropout, training=self.training)
            for ntype, h in h_dict2.items()
        }
        h_dict2 = {
            ntype: h + h_dict1[ntype]  # residual connection
            for ntype, h in h_dict2.items()
        }

        # Type-specific output layer
        out_dict = {
            ntype: self.out_mlp[ntype](h)
            for ntype, h in h_dict2.items()
        }

        return out_dict

class HeteroGCNCiteerDropout(nn.Module):
    def __init__(self, x_dict, hidden_dim, out_dim, metadata, target_feat, dropout=0.5):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.dropout = dropout

        # conv1: in_dim -> hidden_dim for each relation
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(x_dict[target_feat].shape[1], hidden_dim)
            for rel in self.edge_types
        }, aggregate='sum')

        # conv2: hidden_dim -> out_dim for each relation
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hidden_dim, out_dim)
            for rel in self.edge_types
        }, aggregate='sum')

    def forward(self, graph, x_dict):
        h_dict = self.conv1(graph, x_dict)
        h_dict = {ntype: F.relu(h) for ntype, h in h_dict.items()}
        h_dict = {ntype: F.dropout(h, p=self.dropout, training=self.training) for ntype, h in h_dict.items()}
        h_dict = self.conv2(graph, h_dict)
        return h_dict


class HeteroGCNCiteer(nn.Module):
    def __init__(self, x_dict, hidden_dim, out_dim, metadata, target_feat):
        super().__init__()
        # conv1: in_dim -> hidden_dim for each relation
        self.node_types, self.edge_types = metadata
        
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(x_dict[target_feat].shape[1], hidden_dim)
            for rel in self.edge_types
        }, aggregate='sum')
        # conv2: hidden_dim -> out_dim for each relation
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hidden_dim, out_dim)
            for rel in self.edge_types
        }, aggregate='sum')

    def forward(self, graph, x_dict):
        # x_dict: {'paper': feature_tensor}
        h_dict = self.conv1(graph, x_dict)
        # apply activation
        h_dict = {ntype: F.relu(h) for ntype, h in h_dict.items()}
        h_dict = self.conv2(graph, h_dict)
        return h_dict
