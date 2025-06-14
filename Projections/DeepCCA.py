import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.network(x)
class CCA:
    def __init__(self, input_dim_1, input_dim_2, n_components=50, reg=1e-3, lr=1e-3, epochs=50, batch_size=512  , device='cpu'):
        """
        model1, model2: neural networks for view 1 and view 2
        out_dim: output dimensionality for the shared representation
        reg: regularization term for covariance matrices
        """
        
        model1 = MLP(input_dim=input_dim_1, output_dim=n_components)
        model2 = MLP(input_dim=input_dim_2, output_dim=n_components)
        self.model1 = model1.to(device)
        self.model2 = model2.to(device)
        self.out_dim = n_components
        self.reg = reg
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def _cca_loss(self, H1, H2, eps=1e-6):
        """Canonical Correlation Analysis loss for two views."""
        m = H1.size(0)

        H1_bar = H1 - H1.mean(dim=0)
        H2_bar = H2 - H2.mean(dim=0)

        SigmaHat12 = H1_bar.t() @ H2_bar / (m - 1)
        SigmaHat11 = H1_bar.t() @ H1_bar / (m - 1) + self.reg * torch.eye(self.out_dim, device=H1.device)
        SigmaHat22 = H2_bar.t() @ H2_bar / (m - 1) + self.reg * torch.eye(self.out_dim, device=H2.device)

        # matrix square root inverse
        D1_inv = torch.linalg.inv(torch.linalg.cholesky(SigmaHat11))
        D2_inv = torch.linalg.inv(torch.linalg.cholesky(SigmaHat22))

        T = D1_inv @ SigmaHat12 @ D2_inv
        corr = torch.trace(T.t() @ T).sqrt()
        return -corr  # negative because we minimize

    def fit(self, X1, X2):
        """
        Fit the DeepCCA model to the data.

        X1, X2: torch.Tensor datasets (n_samples x input_dim)
        """
        dataset = torch.utils.data.TensorDataset(X1, X2)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()), lr=self.lr)

        self.model1.train()
        self.model2.train()

        for epoch in range(self.epochs):
            total_loss = 0
            for x1, x2 in dataloader:
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)

                z1 = self.model1(x1)
                z2 = self.model2(x2)

                loss = self._cca_loss(z1, z2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")

    def transform(self, X1, X2):
        """
        Transform the data to the shared space.

        Returns the latent representations.
        """
        self.model1.eval()
        self.model2.eval()

        with torch.no_grad():
            Z1 = self.model1(X1)
            Z2 = self.model2(X2)

        return Z1, Z2
