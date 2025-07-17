import torch
from dgl import heterograph
import dgl
import os
import sys
from sklearn.decomposition import PCA
from ogb.lsc import MAG240MDataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../../"))


class MAG:
    def __init__(self):
        pass

    def load_graph(self, n_components=None) -> dgl.DGLGraph:
        dataset = MAG240MDataset(root='./mag240m')
        g = dataset.load_dgl_graph()

        num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}

        # Feature assignment (only paper has features)
        for ntype in g.ntypes:
            if ntype == "paper":
                features = torch.from_numpy(dataset.paper_feat).float()
                g.nodes[ntype].data['feat'] = features

                if n_components is not None:
                    pca = PCA(n_components=n_components)
                    pca_feat = pca.fit_transform((features - features.mean(dim=0)) / (features.std(dim=0) + 1e-4))
                    g.nodes[ntype].data['feat_pca'] = torch.from_numpy(pca_feat).float()
                else:
                    g.nodes[ntype].data['feat_pca'] = features
            else:
                g.nodes[ntype].data['feat'] = torch.ones(num_nodes_dict[ntype], 1)
                g.nodes[ntype].data['feat_pca'] = torch.ones(num_nodes_dict[ntype], 1)

        # Labels and masks (only paper has labels and masks)
        g.nodes['paper'].data['label'] = torch.from_numpy(dataset.paper_label).long()

        split_dict = dataset.get_idx_split()
        g.nodes['paper'].data['train_mask'] = torch.zeros(num_nodes_dict['paper'], dtype=torch.bool)
        g.nodes['paper'].data['valid_mask'] = torch.zeros(num_nodes_dict['paper'], dtype=torch.bool)
        g.nodes['paper'].data['test_mask']  = torch.zeros(num_nodes_dict['paper'], dtype=torch.bool)

        g.nodes['paper'].data['train_mask'][split_dict['train']] = True
        g.nodes['paper'].data['valid_mask'][split_dict['valid']] = True
        g.nodes['paper'].data['test_mask'][split_dict['test']] = True

        self.features = {'paper': torch.from_numpy(dataset.paper_feat).float()}
        self.dataset = dataset
        return g
