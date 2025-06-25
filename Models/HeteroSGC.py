import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, GraphConv
import dgl.function as fn
import torch
from torch_geometric.nn import Linear
import torch.nn.functional as F

class HeteroSGCPaper(torch.nn.Module):
    def __init__(self, metadata, out_dim, hidden_dim , x_dict,target_feat="movie",  num_layers=2 , num_lins=1, alpha=0.01, dropout=0.0):
        super().__init__()
        
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.alpha = alpha
        self.num_layers = num_layers
        self.num_lins = num_lins
        self.target_node_type = target_feat
        self.x_dict = x_dict
        self.in_lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.in_lin_dict[node_type] = torch.nn.ModuleList()
            self.in_lin_dict[node_type].append(Linear(-1, hidden_dim))
            for _ in range(num_lins-1):
                self.in_lin_dict[node_type].append(Linear(hidden_dim, hidden_dim))
        self.out_lin = Linear(hidden_dim, out_dim)
    def reset_parameters(self,init_list=None):
        if init_list is None:
            for node_type in self.in_lin_dict.keys():
                for lin in self.in_lin_dict[node_type]:
                    lin.reset_parameters()
            self.out_lin.reset_parameters()
        else:
            i = 0
            for self_p in self.parameters():
                if self_p.dim()==2:
                    self_p.data.copy_(init_list[i])
                    i += 1
            
    def forward(self, g, x_dict, get_embeddings=False):
        h_dict = {}
        
        # Step 1: Initial MLP stack per node type
        for ntype in x_dict:
            h = self.in_lin_dict[ntype][0](x_dict[ntype]).relu_()
            for lin in self.in_lin_dict[ntype][1:]:
                h = lin(h).relu_()
            h_dict[ntype] = h

        # Register initial node features in the graph
        for ntype, h in h_dict.items():
            g.nodes[ntype].data['h'] = h

        # Step 2: Message passing layers
        for _ in range(self.num_layers):
            out_dict = {ntype: [self.alpha * g.nodes[ntype].data['h']] for ntype in g.ntypes}

            # Use DGL's built-in message passing for all edge types
            funcs = {}
            for etype in g.etypes:
                srctype, _, dsttype = g.to_canonical_etype(etype)
                funcs[etype] = (fn.copy_u('h', 'm'), fn.mean('m', 'm_agg'))

            g.multi_update_all(funcs, cross_reducer='mean')

            # Collect results
            for ntype in g.ntypes:
                if 'm_agg' in g.nodes[ntype].data:
                    out_dict[ntype].append(g.nodes[ntype].data['m_agg'])

            # Sum up contributions
            for ntype in g.ntypes:
                g.nodes[ntype].data['h'] = torch.sum(torch.stack(out_dict[ntype], dim=0), dim=0)

        # Final layer
        target_h = g.nodes[self.target_node_type].data['h']
        target_logits = self.out_lin(target_h)
        out_dict = {}
        out_dict[self.target_node_type]  = target_logits 
        return out_dict
        if get_embeddings:
            embeddings = {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}
            return target_logits, embeddings
        else:
            return target_logits

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
