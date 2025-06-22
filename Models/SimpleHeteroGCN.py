
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, HeteroGraphConv


class HeteroGCN(nn.Module):
    def __init__(self, x_dict, hidden_dim, out_dim, metadata, dropout=0.5):
        super().__init__()

        self.layer1 = HeteroGraphConv({
            rel: GraphConv(x_dict[rel], hidden_dim)
            for rel in metadata[1]
        })

        self.layer2 = HeteroGraphConv({
            rel: GraphConv(hidden_dim, out_dim)
            for rel in metadata[1]
        })

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, inputs):
        """
        :param g: DGLHeteroGraph
        :param inputs: Dict of node features per node type {ntype: tensor}
        :return: Dict of updated node features {ntype: tensor}
        """
        h = self.layer1(g, inputs)
        h = {k: F.relu(self.dropout(v)) for k, v in h.items()}
        h = self.layer2(g, h)
        return h
    
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
