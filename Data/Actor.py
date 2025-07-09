import dgl
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from dgl.data import Actor

class Actor():
    def __init__(self):
        pass

    def load_graph(self, n_components=50) -> dgl.DGLGraph:
        """
        Load the homogeneous Citeseer citation graph and apply feature reduction,
        transformation, and normalization. This version can work for actor datasets
        with appropriate adaptations for node feature manipulations.

        Parameters:
        ----------
        n_components : int, optional
            The number of principal components to reduce the feature dimension to.
            Default is 50.

        Returns:
        -------
        dgl.DGLGraph
            A DGL heterogeneous graph object with reduced and normalized node features.
        """
        # Load the Citeseer dataset graph
        dataset = Actor()
        g = dataset[0]

        # Original node features (e.g., word counts, embeddings, etc.)
        data = g.ndata['feat']  # shape: (num_nodes, orig_feat_dim)

        # Reduce to n_components dimensions using PCA (if specified)
        if n_components:
            reduced_feat = self.reduce_features(data, n_components=n_components)
        else:
            reduced_feat = data

        # Build the heterogeneous graph with citation relations
        src, dst = g.edges()
        data_dict = {
            ('paper', 'cites', 'paper'): (src, dst),
        }
        hetero_g = dgl.heterograph(data_dict)

        # Normalize the features between 0 and 1 using MinMaxScaler
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(reduced_feat.numpy())
        
        # Assign reduced and normalized features to the graph
        hetero_g.nodes['paper'].data['feat'] = torch.tensor(normalized_features, dtype=torch.float32)
        hetero_g.nodes['paper'].data['feat_pca'] = torch.tensor(reduced_feat.numpy(), dtype=torch.float32)

        # Assign labels and masks (training, validation, test)
        hetero_g.nodes['paper'].data['label'] = g.ndata['label']
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
        # Convert tensor to numpy array for PCA
        X = feat_tensor.numpy()
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        
        # Convert reduced features back to a PyTorch tensor
        return torch.tensor(X_reduced, dtype=torch.float32)
