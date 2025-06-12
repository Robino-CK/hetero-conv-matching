
import torch
import torch.nn as nn
import torch.nn.functional as F

class NonlinearStochasticCCA(nn.Module):
    def __init__(
        self, input_dim1, input_dim2, n_components=10,
        hidden_dims=(128, 64), dropout=0.3,
        lr=1e-3, weight_decay=1e-4, device='cpu'
    ):
        super().__init__()
        self.latent_dim = n_components
        self.device = device

        self.net1 = self._build_mlp(input_dim1, hidden_dims, n_components, dropout).to(device)
        self.net2 = self._build_mlp(input_dim2, hidden_dims, n_components, dropout).to(device)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _build_mlp(self, input_dim, hidden_dims, output_dim, dropout):
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        return nn.Sequential(*layers)

    def _normalize(self, x):
        return F.normalize(x, dim=0)

    def loss(self, z1, z2):
        z1 = self._normalize(z1)
        z2 = self._normalize(z2)
        return -torch.mean(torch.sum(z1 * z2, dim=1))

    def partial_fit(self, x_batch, y_batch):
        self.train()
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        z1 = self.net1(x_batch)
        z2 = self.net2(y_batch)

        loss = self.loss(z1, z2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def transform(self, x, y):
        self.eval()
        with torch.no_grad():
            x_proj = self.net1(x.to(self.device))
            y_proj = self.net2(y.to(self.device))
        return x_proj, y_proj
    def fit(self, X, Y):
        batch_size = 64
        for epoch in range(150):
      #      perm = torch.randperm(X.size(0))
        #    for i in range(0, X.size(0), batch_size):
       #         idx = perm[i:i+batch_size]
        #        x_batch = X[idx]
         #       y_batch = Y[idx]
          #      loss = self.partial_fit(x_batch, y_batch)
            loss = self.partial_fit(X, Y)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
