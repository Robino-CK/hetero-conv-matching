import torch
from dgl import heterograph
import dgl
import torch_geometric
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler
import dgl
import os
import sys
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.graph_store import EdgeAttr
from torch_geometric.data.storage import NodeStorage
from torch_geometric.data.storage import BaseStorage
from torch_geometric.data.storage import EdgeStorage
import torch
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../../"))


class IMDB():
    def __init__(self):
        pass    
        #super().__init__(name='Cora')

   
   #     self.epochs = 600

    def load_graph(self, n_components=100) -> dgl.DGLGraph:
        # torch.serialization.add_safe_globals([HeteroData])
        # torch.serialization.add_safe_globals([HeteroData])
        # torch.serialization.add_safe_globals([EdgeAttr])
        # torch.serialization.add_safe_globals([torch_geometric.data.TensorAttr])
        # torch.serialization.add_safe_globals([BaseStorage])
        # torch.serialization.add_safe_globals([NodeStorage])
        # torch.serialization.add_safe_globals([EdgeStorage])
        
        # Load the dataset
        path = './imdb'
        dataset = torch_geometric.datasets.IMDB(path)
        data = dataset[0]  # Only one graph

        num_nodes_dict = {
            'movie': data.x_dict['movie'].shape[0],
            'director': data.x_dict['director'].shape[0],
            'actor': data.x_dict['actor'].shape[0],
        #    'conference': data["conference"]["num_nodes"]
        }

        # Extract edge index dict from torch geometric
        edge_index_dict = data.edge_index_dict

        # Convert PyG edge index dict to DGL edge tuples
        edge_tuples = {}
        
        for (src_type, rel_type, dst_type), edge_index in edge_index_dict.items():
            # if src_type == "movie":
            #     continue
            src_nodes = edge_index[0].numpy()
            dst_nodes = edge_index[1].numpy()
            edge_tuples[(src_type, src_type + rel_type + dst_type, dst_type)] = (src_nodes, dst_nodes)
        # Create DGL heterograph
        g = heterograph(edge_tuples, num_nodes_dict=num_nodes_dict)
        
        # You can also assign features
        for ntype in g.ntypes:
            if n_components == None:
                    g.nodes[ntype].data['feat_pca'] =data.x_dict[ntype]
            else:
                pca = PCA(n_components=n_components)
                
                pca_feat = pca.fit_transform((data.x_dict[ntype] - data.x_dict[ntype].mean(dim=0)) / (data.x_dict[ntype].std(dim=0) + 0.0001))
               # scaler = MinMaxScaler()

                # Normalize the features between 0 and 1
                #normalized_features = scaler.fit_transform(pca_feat)
                g.nodes[ntype].data['feat_pca'] = torch.from_numpy(pca_feat).type(torch.FloatTensor)
            g.nodes[ntype].data['feat'] = data.x_dict[ntype]
        g.nodes["movie"].data['label'] = data["movie"].y
        for key in ['train_mask', 'val_mask', 'test_mask']:
            g.nodes['movie'].data[key] = data["movie"][key]

        self.features = data.x_dict
        self.dataset = data
        
        for key in ['train_mask', 'val_mask', 'test_mask']:
            g.nodes['movie'].data[key] = data["movie"][key]

        return g

