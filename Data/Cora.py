import dgl
import torch
import torch.nn.functional as F
from dgl.data import CoraGraphDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class Cora():
    def __init__(self):
        pass

    def load_graph(self, n_components=50) -> dgl.DGLGraph:
        # Load the homogeneous Cora citation graph
        dataset = CoraGraphDataset()
        g = dataset[0]

        # Original node features
        data = g.ndata['feat']  # shape: (num_nodes, orig_feat_dim)

        # Reduce to n_components if requested
        if n_components:
            reduced_feat = self.reduce_features(data, n_components=n_components)
        else:
            reduced_feat = data

        # Build heterogeneous graph with 'cites' relation
        src, dst = g.edges()
        data_dict = {
            ('paper', 'cites', 'paper'): (src, dst),
            # ('paper', 'cited-by', 'paper'): (dst, src),  # Optional reverse edge
        }
        hetero_g = dgl.heterograph(data_dict)

        # Normalize the reduced features
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(reduced_feat)

        # Assign features and labels
        hetero_g.nodes['paper'].data['feat'] = data  # original
        hetero_g.nodes['paper'].data['feat_pca'] = reduced_feat  # reduced
        hetero_g.nodes['paper'].data['label'] = g.ndata['label']

        # Assign masks
        for key in ['train_mask', 'val_mask', 'test_mask']:
            hetero_g.nodes['paper'].data[key] = g.ndata[key]

        return hetero_g

    def reduce_features(self, feat_tensor, n_components=50):
        """
        Reduce feature dimension using PCA.

        Parameters
        ----------
        feat_tensor : torch.Tensor
            Input node features of shape (N, D).
        n_components : int
            Target number of dimensions.

        Returns
        -------
        torch.Tensor
            Transformed features of shape (N, n_components).
        """
        X = feat_tensor.numpy()
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        return torch.from_numpy(X_reduced).float()
