import torch
from dgl import heterograph
import dgl
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.datasets import HGBDataset
from torch_geometric.data import HeteroData
from torch_geometric.data.graph_store import EdgeAttr
from torch_geometric.data.storage import NodeStorage, BaseStorage, EdgeStorage

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../../"))


class ACM():
    def __init__(self):
        pass

    def load_graph(self, n_components) -> dgl.DGLGraph:
        path = './acm'
        dataset = HGBDataset(path,  name='ACM')
        data = dataset[0]  # Only one HeteroData object

        num_nodes_dict = {
            'author': data.x_dict['author'].shape[0],
            'paper': data.x_dict['paper'].shape[0],
            'subject': data.x_dict['subject'].shape[0],
            'term': data["term"]["num_nodes"]
        }

        # Convert edge_index_dict from PyG to DGL
        edge_index_dict = data.edge_index_dict
        edge_tuples = {}
        for (src_type, rel_type, dst_type), edge_index in edge_index_dict.items():
            src_nodes = edge_index[0].numpy()
            dst_nodes = edge_index[1].numpy()
            edge_tuples[(src_type, f"{src_type}to{dst_type}", dst_type)] = (src_nodes, dst_nodes)

        g = heterograph(edge_tuples, num_nodes_dict=num_nodes_dict)

        # Feature assignment with optional PCA
        for ntype in g.ntypes:
            if ntype == "term":
                g.nodes[ntype].data['feat'] = torch.ones(num_nodes_dict['term'],1)
                g.nodes[ntype].data['feat_pca'] = torch.ones(num_nodes_dict['term'],1)    
                continue
            features = data.x_dict[ntype]

            if n_components is None:
                g.nodes[ntype].data['feat_pca'] = features
            else:
                pca = PCA(n_components=n_components)
                pca_feat = pca.fit_transform((features - features.mean(dim=0)) / (features.std(dim=0) + 1e-4))
                g.nodes[ntype].data['feat_pca'] = torch.from_numpy(pca_feat).float()

            g.nodes[ntype].data['feat'] = features

        # Labels and masks (only author has labels in this dataset)
        g.nodes["paper"].data['label'] = data["paper"].y
        for mask_name in ['train_mask',  'test_mask']:
            g.nodes['paper'].data[mask_name] = data['paper'][mask_name]

        self.features = data.x_dict
        self.dataset = data
        return g
