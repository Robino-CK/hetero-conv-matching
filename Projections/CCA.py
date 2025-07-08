import torch
import torch.nn as nn
import torch.nn.functional as F



class CCAOld:
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


import torch
import torch.linalg as LA

class CCA:
    def __init__(self, x, y, n_components=None, reg=1e-6, device="cpu"):
        """
        Initialize CCA model.
        
        Args:
            latent_dims (int): number of canonical components to keep.
        """
        self.device = device
        self.latent_dims = n_components
        self.Wx = None  # projection matrix for X
        self.Wy = None  # projection matrix for Y
        self.mean_x = None
        self.mean_y = None
        self.name = 'cca'
        
    def fit(self, X, Y):
        """
        Fit CCA model to views X and Y.
        
        Args:
            X (torch.Tensor): first view, shape (n_samples, n_features_x)
            Y (torch.Tensor): second view, shape (n_samples, n_features_y)
        """
        n, dx = X.shape
        _, dy = Y.shape
        assert n == Y.shape[0], "Number of samples must be the same for X and Y"
        
        # Center the data
        self.mean_x = torch.mean(X, dim=0, keepdim=True)
        self.mean_y = torch.mean(Y, dim=0, keepdim=True)
        Xc = X - self.mean_x
        Yc = Y - self.mean_y
        
        # Covariance matrices
        Sxx = (Xc.T @ Xc) / (n - 1) + 1e-8 * torch.eye(dx, device=self.device)  # regularization for numerical stability
        Syy = (Yc.T @ Yc) / (n - 1) + 1e-8 * torch.eye(dy, device=self.device)
        Sxy = (Xc.T @ Yc) / (n - 1)
        
        # Whitening transformations
        # Compute inverse sqrt of covariance matrices using eigen decomposition
        eigvals_x, eigvecs_x = LA.eigh(Sxx)
        eigvals_y, eigvecs_y = LA.eigh(Syy)
        
        # Inverse sqrt of covariance
        Sxx_inv_sqrt = eigvecs_x @ torch.diag(eigvals_x.clamp(min=1e-12).pow(-0.5)) @ eigvecs_x.T
        Syy_inv_sqrt = eigvecs_y @ torch.diag(eigvals_y.clamp(min=1e-12).pow(-0.5)) @ eigvecs_y.T
        
        # Compute matrix for SVD
        T = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
        
        # SVD
        U, S, Vh = torch.linalg.svd(T)
        
        # Select the top latent_dims components
        U = U[:, :self.latent_dims]
        V = Vh.T[:, :self.latent_dims]
        
        # Projection matrices
        self.Wx = Sxx_inv_sqrt @ U
        self.Wy = Syy_inv_sqrt @ V
    
    def quality(self, X, Y):
        X_proj, Y_proj = self.transform(X,Y)
        x = X_proj[:, 0]
        y = Y_proj[:, 0]
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        corr = (x_centered @ y_centered) / (
            x_centered.norm() * y_centered.norm()
        )
        return corr

    def transform(self, X=None, Y=None):
        """
        Project X and/or Y to the canonical components.
        
        Args:
            X (torch.Tensor or None): first view, shape (n_samples, n_features_x)
            Y (torch.Tensor or None): second view, shape (n_samples, n_features_y)
        
        Returns:
            Tuple of projected views. If one of X or Y is None, returns only the transformed one.
            - X_proj: shape (n_samples, latent_dims) or None
            - Y_proj: shape (n_samples, latent_dims) or None
        """
        X_proj, Y_proj = None, None
        
        if X is not None:
            Xc = X - self.mean_x
            X_proj = Xc @ self.Wx
        if Y is not None:
            Yc = Y - self.mean_y
            Y_proj = Yc @ self.Wy
        
        if X_proj is not None and Y_proj is not None:
            
            return X_proj, Y_proj
        elif X_proj is not None:
            return X_proj
        else:
            return Y_proj
