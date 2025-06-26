import torch
import torch.nn as nn
import torch.nn.functional as F

class ViewEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, latent_dim)
        self.to(self.device)
        
    def forward(self, x):
        x = x.to(self.device)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ViewDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.to(self.device)
        
    def forward(self, z):
        z = z.to(self.device)
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        return z

class MultiViewAutoencoder(nn.Module):
    def __init__(self, input_dims, latent_dim, device=None):
        """
        input_dims: list of ints, each is d_i1 * d_i2 for the i-th view
        latent_dim: int, dimension of shared latent space per view
        device: torch.device or None, device to move model to (e.g., 'cuda' or 'cpu')
        """
        super().__init__()
        self.num_views = len(input_dims)
        self.device = device if device is not None else torch.device('cpu')
        
        # Create one encoder and decoder per view
        self.encoders = nn.ModuleList([ViewEncoder(dim, latent_dim, device) for dim in input_dims])
        self.decoders = nn.ModuleList([ViewDecoder(latent_dim, dim, device) for dim in input_dims])
        
        # Move model to device
        self.to(self.device)
        
    def forward(self, *inputs):
        assert len(inputs) == self.num_views, "Number of inputs must match number of views"
        
        # Move inputs to device
        inputs = [x.to(self.device) for x in inputs]
        
        latents = [enc(x) for enc, x in zip(self.encoders, inputs)]
        
        joint_latent = torch.cat(latents, dim=1)
        
        reconstructions = []
        start = 0
        for i in range(self.num_views):
            end = start + latents[i].shape[1]  # latent_dim
            z_i = joint_latent[:, start:end]
            out = self.decoders[i](z_i)
            out = out.view(inputs[i].shape)
            reconstructions.append(out)
            start = end
        
        return reconstructions

    def encode(self, *inputs):
        inputs = [x.to(self.device) for x in inputs]
        latents = [enc(x) for enc, x in zip(self.encoders, inputs)]
        return latents  # list of latent tensors, one per view
