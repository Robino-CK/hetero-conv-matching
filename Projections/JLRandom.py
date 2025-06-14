import torch
import torch.nn as nn
from torch.nn import Linear

class JLRandomProjection(nn.Module):
    def __init__(self, in_dim, out_dim, method='sparse', bias=False, device='cpu'):
        """
        JL-style random projection layer.
        
        method: 'gaussian', 'rademacher', or 'sparse'
        device: device for the projection matrix (e.g., 'cuda' or 'cpu')
        """
        super(JLRandomProjection, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=bias).to(device)

        with torch.no_grad():
            if method == 'gaussian':
                W = torch.randn(out_dim, in_dim, device=device) / (out_dim ** 0.5)
            elif method == 'rademacher':
                W = torch.randint(0, 2, (out_dim, in_dim), device=device, dtype=torch.float32)
                W[W == 0] = -1.0
                W /= (out_dim ** 0.5)
            elif method == 'sparse':
                W = torch.zeros(out_dim, in_dim, device=device)
                s = 3
                for i in range(out_dim):
                    idx = torch.randperm(in_dim, device=device)[:s]
                    vals = torch.randint(0, 2, (s,), device=device) * 2 - 1
                    W[i, idx] = vals / (s ** 0.5)
            else:
                raise ValueError("Unsupported method: choose 'gaussian', 'rademacher', or 'sparse'.")

            self.proj.weight.data = W
            self.proj.weight.requires_grad = False

            if bias:
                self.proj.bias.data.zero_()
                self.proj.bias.requires_grad = False

    def forward(self, x):
        return self.proj(x)
