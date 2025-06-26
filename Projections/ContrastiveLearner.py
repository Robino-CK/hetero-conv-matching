import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ContrastiveLearner:
    def __init__(self, input_dim_1, input_dim_2, n_components=128, temperature=0.5, lr=1e-3, epochs=1000, batch_size=512, device='cpu'):
        out_dim = n_components
        self.model1 = MLP(input_dim=input_dim_1, output_dim=out_dim).to(device)
        self.model2 = MLP(input_dim=input_dim_2, output_dim=out_dim).to(device)
        self.out_dim = out_dim
        self.temperature = temperature
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def _nt_xent_loss(self, z1, z2):
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
        z1, z2: tensors of shape (batch_size, out_dim)
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        representations = torch.cat([z1, z2], dim=0)  # (2N, D)
        similarity_matrix = torch.matmul(representations, representations.T)  # (2N, 2N)

        # remove similarity with itself
        mask = torch.eye(similarity_matrix.size(0), device=self.device).bool()
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.size(0), -1)

        positives = torch.sum(z1 * z2, dim=-1)
        positives = torch.cat([positives, positives], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = torch.sum(torch.exp(similarity_matrix / self.temperature), dim=-1)

        loss = -torch.log(nominator / denominator).mean()
        return loss

    def fit(self, X1, X2):
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

                loss = self._nt_xent_loss(z1, z2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")

    def transform(self, X1, X2):
        self.model1.eval()
        self.model2.eval()
        with torch.no_grad():
            Z1 = self.model1(X1.to(self.device))
            Z2 = self.model2(X2.to(self.device))
        return Z1, Z2
