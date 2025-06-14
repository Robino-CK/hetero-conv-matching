import torch
import torch.nn as nn
import torch.nn.functional as F



class CCA:
    def __init__(self, x, y, n_components=None, reg=1e-6, device="cpu"):
        """
        Canonical Correlation Analysis (CCA) using PyTorch.

        Parameters:
        - out_dim: number of canonical components to keep (if None, keep all).
        - reg: regularization parameter (for numerical stability).
        """
        self.out_dim = n_components
        self.reg = reg
        self.Wx = None
        self.Wy = None
        self.mean_x = None
        self.mean_y = None

    def fit(self, X, Y):
        """
        Learn CCA projection matrices from paired datasets X and Y.

        Parameters:
        - X: torch.Tensor of shape (n_samples, n_features_x)
        - Y: torch.Tensor of shape (n_samples, n_features_y)
        """
        # Center the data
        self.mean_x = X.mean(dim=0)
        self.mean_y = Y.mean(dim=0)
        Xc = X - self.mean_x
        Yc = Y - self.mean_y

        n = X.shape[0]

        # Covariance matrices
        Cxx = (Xc.T @ Xc) / (n - 1) + self.reg * torch.eye(X.shape[1], device=X.device)
        Cyy = (Yc.T @ Yc) / (n - 1) + self.reg * torch.eye(Y.shape[1], device=Y.device)
        Cxy = (Xc.T @ Yc) / (n - 1)

        # Solve generalized eigenvalue problem
        Cxx_inv = torch.linalg.inv(Cxx)
        Cyy_inv = torch.linalg.inv(Cyy)

        eigvals, Wx = torch.linalg.eigh(Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T)
        idx = torch.argsort(eigvals, descending=True)
        Wx = Wx[:, idx]

        Wy = torch.linalg.inv(Cyy) @ Cxy.T @ Wx

        if self.out_dim is not None:
            Wx = Wx[:, :self.out_dim]
            Wy = Wy[:, :self.out_dim]

        self.Wx = Wx
        self.Wy = Wy

    def transform(self, X, Y=None):
        """
        Project X (and optionally Y) onto the canonical components.

        Parameters:
        - X: torch.Tensor of shape (n_samples, n_features_x)
        - Y: torch.Tensor of shape (n_samples, n_features_y), optional

        Returns:
        - Zx: CCA projection of X
        - Zy: CCA projection of Y (if Y is provided)
        """
        Xc = X - self.mean_x
        Zx = Xc @ self.Wx
        if Y is not None:
            Yc = Y - self.mean_y
            Zy = Yc @ self.Wy
            return Zx, Zy
        return Zx
