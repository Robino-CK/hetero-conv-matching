import dgl
import os
import sys

import torch.nn.functional

from sklearn.preprocessing import MinMaxScaler, StandardScaler 

#from Datasets.NodeClassification.NodeClassificationDataset import NodeClassificationDataset

from dgl.data import CiteseerGraphDataset
import torch
from sklearn.decomposition import PCA

class Citeseer():
    def __init__(self):
        pass

    def load_graph(self, n_components=50) -> dgl.DGLGraph:
        # Load the homogeneous Citeseer citation graph
        dataset = CiteseerGraphDataset()
        g = dataset[0]

        # Original node features
        data = g.ndata['feat']  # shape: (num_nodes, orig_feat_dim)


        # Reduce to 50 dimensions (you can adjust this)
        if n_components:
            # scaler = StandardScaler()
            # scaler.fit(data)
            # scaled_data = scaler.transform(data) 
            # scaled_data = torch.tensor(scaled_data, dtype=torch.double)
            reduced_feat = self.reduce_features(data, n_components=n_components)
        else:
            reduced_feat = data
        # Build heterogeneous graph with 'cites' and 'cited-by' relations
        src, dst = g.edges()
        data_dict = {
            ('paper', 'cites', 'paper'): (src, dst),
    #        ('paper', 'cited-by', 'paper'): (dst, src)
        }
        hetero_g = dgl.heterograph(data_dict)
        scaler = MinMaxScaler()

            # Normalize the features between 0 and 1
        normalized_features = scaler.fit_transform(reduced_feat)
        
        # Assign reduced features and other data
        hetero_g.nodes['paper'].data['feat'] = data#torch.from_numpy(normalized_features).type(torch.FloatTensor)
        
        hetero_g.nodes['paper'].data['feat_pca'] = reduced_feat#torch.from_numpy(normalized_features).type(torch.FloatTensor)
        hetero_g.nodes['paper'].data['label'] = g.ndata['label']
        for key in ['train_mask', 'val_mask', 'test_mask']:
            hetero_g.nodes['paper'].data[key] = g.ndata[key]

        return hetero_g
    
    # Apply PCA to reduce feature dimension
    # Choose the number of principal components
    def reduce_features(self,feat_tensor, n_components=50):
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
        # Convert to numpy for sklearn PCA
        X = feat_tensor.numpy()
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        return torch.from_numpy(X_reduced).float()