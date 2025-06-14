import torch
import torch.nn as nn
import torch.nn.functional as F

class StochasticCCA:
    def __init__(self,  input_dim1, input_dim2,n_components=50, lr=1e-3, device='cpu'):
        self.latent_dim = n_components
        self.device = device

        # Linear mappings to latent space
        self.Wx = torch.nn.Linear(input_dim1, n_components, bias=False).to(device)
        self.Wy = torch.nn.Linear(input_dim2, n_components, bias=False).to(device)

        self.optimizer = torch.optim.Adam(list(self.Wx.parameters()) + list(self.Wy.parameters()), lr=lr)

    def _center(self, x):
        return x - x.mean(dim=0, keepdim=True)

    def _normalize(self, x):
        return F.normalize(x, dim=0)

    def loss(self, x_proj, y_proj):
        """
        Negative correlation objective (maximize correlation)
        """
        x_proj = self._normalize(x_proj)
        y_proj = self._normalize(y_proj)
        return -torch.mean(torch.sum(x_proj * y_proj, dim=1))

    def fit(self, X, Y):
        batch_size = 64
        for epoch in range(300):
      #      perm = torch.randperm(X.size(0))
        #    for i in range(0, X.size(0), batch_size):
       #         idx = perm[i:i+batch_size]
        #        x_batch = X[idx]
         #       y_batch = Y[idx]
          #      loss = self.partial_fit(x_batch, y_batch)
            loss = self.partial_fit(X, Y)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
        
        
    def partial_fit(self, x_batch, y_batch):
        """
        One step of training using a mini-batch.
        """
        self.Wx.train()
        self.Wy.train()

        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        x_proj = self.Wx(x_batch)
        y_proj = self.Wy(y_batch)

        loss = self.loss(x_proj, y_proj)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def transform(self, x, y):
        """
        Projects full datasets to the shared latent space.
        """
        self.Wx.eval()
        self.Wy.eval()
        with torch.no_grad():
            x_latent = self.Wx(x.to(self.device))
            y_latent = self.Wy(y.to(self.device))
        return x_latent, y_latent
