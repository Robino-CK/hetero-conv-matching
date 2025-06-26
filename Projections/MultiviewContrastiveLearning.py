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

class MultiViewContrastiveLearner:
    def __init__(self, input_dims, n_components=128, temperature=0.5, lr=1e-3, epochs=300, batch_size=512, device='cpu'):
        """
        input_dims: list of input dimensions for each view, e.g. [input_dim_1, input_dim_2, ..., input_dim_K]
        """
        self.device = device
        self.n_views = len(input_dims)
        self.models = nn.ModuleList([MLP(input_dim=d, output_dim=n_components).to(device) for d in input_dims])
        self.out_dim = n_components
        self.temperature = temperature
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def _nt_xent_loss(self, zs):
        """
        NT-Xent loss generalized for multiple views.
        zs: list of normalized embeddings, one per view, each (batch_size, out_dim)
        """
        batch_size = zs[0].size(0)
        out_dim = zs[0].size(1)
        n_views = len(zs)

        # Normalize embeddings
        zs = [F.normalize(z, dim=1) for z in zs]

        # Concatenate all views embeddings into one matrix (K * N, D)
        representations = torch.cat(zs, dim=0)

        # Compute similarity matrix (K*N, K*N)
        similarity_matrix = torch.matmul(representations, representations.T)

        # Mask to remove similarity of samples to themselves
        mask = torch.eye(n_views * batch_size, device=self.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)  # very large negative number for masking

        loss = 0.0
        # For each view pair (i,j), i != j, treat zs[i][k] and zs[j][k] as positive pairs
        for i in range(n_views):
            for j in range(n_views):
                if i == j:
                    continue
                # Positive similarity: diagonal elements between view i and view j embeddings
                positives = torch.sum(zs[i] * zs[j], dim=1)  # (batch_size,)

                # For each sample in batch, calculate loss
                numerator = torch.exp(positives / self.temperature)

                # denominator: similarity of all pairs for embeddings from view i
                # rows for view i embeddings: slice from i*batch_size to (i+1)*batch_size
                sim_i = similarity_matrix[i*batch_size:(i+1)*batch_size, :]

                denominator = torch.sum(torch.exp(sim_i / self.temperature), dim=1)

                loss_ij = -torch.log(numerator / denominator).mean()
                loss += loss_ij

        # Average over number of pairs (K*(K-1))
        loss /= (n_views * (n_views - 1))
        return loss

    def fit(self, *Xs):
        """
        Xs: variable number of input tensors, one per view, shape (N, input_dim_i)
        """
        assert len(Xs) == self.n_views, f"Expected {self.n_views} views, got {len(Xs)}"

        dataset = torch.utils.data.TensorDataset(*Xs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.models.parameters(), lr=self.lr)

        for model in self.models:
            model.train()

        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                batch = [x.to(self.device) for x in batch]

                zs = [self.models[i](batch[i]) for i in range(self.n_views)]

                loss = self._nt_xent_loss(zs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")

    def transform(self, *Xs):
        """
        Returns embeddings from all views.
        """
        assert len(Xs) == self.n_views, f"Expected {self.n_views} views, got {len(Xs)}"

        for model in self.models:
            model.eval()

        with torch.no_grad():
            zs = [self.models[i](Xs[i].to(self.device)) for i in range(self.n_views)]

        return zs
