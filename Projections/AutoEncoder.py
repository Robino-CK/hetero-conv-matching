import torch
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

class MultiviewAutoencoder:
    def __init__(self, input_dim_1, input_dim_2, n_components=50, lr=1e-3, epochs=500, batch_size=512, align_views=True, device='cpu'):
        self.encoder1 = MLP(input_dim_1, n_components).to(device)
        self.encoder2 = MLP(input_dim_2, n_components).to(device)
        self.decoder1 = MLP(n_components, input_dim_1).to(device)
        self.decoder2 = MLP(n_components, input_dim_2).to(device)

        self.latent_dim = n_components
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.align_views = align_views  # whether to include alignment loss
        self.name = 'auto'

    def _loss(self, x1, x2, x1_recon, x2_recon, z1, z2):
        # Reconstruction loss (MSE)
        loss_recon_1 = nn.functional.mse_loss(x1_recon, x1)
        loss_recon_2 = nn.functional.mse_loss(x2_recon, x2)
        loss = loss_recon_1 + loss_recon_2

        # Optional alignment loss (cosine similarity)
        if self.align_views:
            cos_sim = nn.functional.cosine_similarity(z1, z2, dim=-1)
            loss_align = 1 - cos_sim.mean()  # maximize similarity
            loss += loss_align

        return loss

    def fit(self, X1, X2):
        dataset = torch.utils.data.TensorDataset(X1, X2)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        params = list(self.encoder1.parameters()) + list(self.encoder2.parameters()) + \
                 list(self.decoder1.parameters()) + list(self.decoder2.parameters())

        optimizer = optim.Adam(params, lr=self.lr)

        for epoch in range(self.epochs):
            total_loss = 0
            for x1, x2 in dataloader:
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)

                z1 = self.encoder1(x1)
                z2 = self.encoder2(x2)

                x1_recon = self.decoder1(z1)
                x2_recon = self.decoder2(z2)

                loss = self._loss(x1, x2, x1_recon, x2_recon, z1, z2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")

    def transform(self, X1=None, X2=None):
        self.encoder1.eval()
        self.encoder2.eval()
        with torch.no_grad():
            
            if X2 == None:
                return self.encoder1(X1.to(self.device))
            if X1 == None:
                return self.encoder2(X2.to(self.device))
            Z1 = self.encoder1(X1.to(self.device))
            Z2 = self.encoder2(X2.to(self.device))
        return Z1, Z2

    def reconstruct(self, X1, X2):
        self.encoder1.eval()
        self.encoder2.eval()
        self.decoder1.eval()
        self.decoder2.eval()
        with torch.no_grad():
            Z1 = self.encoder1(X1.to(self.device))
            Z2 = self.encoder2(X2.to(self.device))
            X1_recon = self.decoder1(Z1)
            X2_recon = self.decoder2(Z2)
        return X1_recon, X2_recon
