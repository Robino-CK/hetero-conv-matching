
import torch
import dgl
from Coarsener.HeteroCoarsener import HeteroCoarsener
import time
import dgl.backend as F         # this is the backend (torch, TF, etc.)
from torch_scatter import scatter_add   
import numpy as np

import torch

class CCALin:
    def __init__(self, n_components=None, reg=1e-6):
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

import torch
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

class HeteroRGCNCoarsener(HeteroCoarsener):
    
    def _get_back_etype(self, etype):
        for src_type, etype_orig, dst_type in self.summarized_graph.canonical_etypes:
            if etype_orig == etype:
                return f"{dst_type}to{src_type}"
    
    def _create_sgn_layer(self, k =1):
        for src_type, etype, _ in self.summarized_graph.canonical_etypes:
            if not self.multi_relations:
                g = dgl.add_self_loop(self.summarized_graph, etype=etype)
            else:
                g = self.summarized_graph
            if self.device == "cpu":
                A = g.adj_external(etype=etype).to_dense()
            else:
                A = g.adj(etype=etype).to_dense()
                
            D = torch.diag(torch.rsqrt(torch.sum(A, dim=1)))
            feat =  self.summarized_graph.nodes[src_type].data['feat']
            H =  torch.pow((D @A @  D), k ) @ feat
            self.summarized_graph.nodes[src_type].data[f"SGC{etype}"] = H
       
        
    def _create_gnn_layer(self):
        
        """
        Vectorized GPU implementation of spatial RGCN coarsening.
        Returns H and S dicts with per-etype tensors.
        """
        
        start_time = time.time()

        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            # Determine if we have node features
            has_feat = 'feat' in self.summarized_graph.nodes[dst_type].data

            # Precompute normalized degrees
            print(etype)
            deg_out = self.original_graph.out_degrees(etype= etype)
            n_src =  self.summarized_graph.num_nodes(src_type)
            n_dst =  self.summarized_graph.num_nodes(dst_type)
            c = self.summarized_graph.nodes[src_type].data['node_size']
            
            inv_sqrt_out = torch.rsqrt(deg_out + self.summarized_graph.nodes[src_type].data['node_size'])
            inv_sqrt_in = torch.rsqrt(self.summarized_graph.in_degrees(etype= etype) + self.summarized_graph.nodes[dst_type].data['node_size'])

            # Load features or use scalar 1
            if has_feat:
                feats = self.summarized_graph.nodes[dst_type].data['feat'].to(self.device)
                feat_dim = feats.shape[1]
            else:
                # treat feature as scalar 1
                feat_dim = 1

            # Extract all edges of this type
            
            u, v = self.summarized_graph.edges(etype=(src_type, etype, dst_type))
            u = u.to(self.device)
            v = v.to(self.device)

            # Gather destination feats & normalize
            if has_feat:
                feat_v = feats[v]                              # [E, D]
            else:
                feat_v = torch.ones((v.shape[0], 1), device=self.device)
                
            if self.use_out_degree:
                s_e = feat_v * inv_sqrt_out[v].unsqueeze(-1)       # [E, D]
            else:
                s_e = feat_v * inv_sqrt_in[v].unsqueeze(-1)
            # Scatter-add to compute S at source nodes
           
            S_tensor = torch.zeros((n_src, feat_dim), device=self.device)
            S_tensor = S_tensor.index_add(0, u, s_e)
            infl = torch.zeros(n_dst, device=self.device)
            if self.approx_neigh:
                infl = infl.index_add(0, v, inv_sqrt_out[u])
                if self.multi_relations:
                    etype_around = f"{dst_type}to{src_type}"
                else:
                    etype_around = etype
                self.summarized_graph.nodes[dst_type].data[f'i{etype_around}'] = infl
    
            # Compute H = D_out^{-1/2} * S
            H_tensor = inv_sqrt_out.unsqueeze(-1) * S_tensor

            # Store in summarized_graph
            
            self.summarized_graph.nodes[src_type].data[f's{etype}'] = S_tensor
            if self.feat_in_gcn:
                self.summarized_graph.nodes[src_type].data[f'h{etype}'] = H_tensor + ((feats / (deg_out + c).unsqueeze(1) ))
            else:
                self.summarized_graph.nodes[src_type].data[f'h{etype}'] = H_tensor
           
           
        if self.use_cca:
            self.ccas = dict()
            for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
                feat_src =   self.summarized_graph.nodes[src_type].data['feat']
                h_src = self.summarized_graph.nodes[src_type].data[f'h{etype}']
                
                cca = StochasticCCA(feat_src.shape[1], h_src.shape[1], n_components=feat_src.shape[1], device=self.device)
                cca.fit(feat_src, h_src) 
                self.ccas[etype] = cca
                
     #   print("_create_h_spatial_rgcn", time.time() - start_time)

    
    def _create_h_merged(self,  node1s, node2s,ntype, etype):
        cache = self.summarized_graph.nodes[ntype].data[f's{etype}']
        
        # 1) Flatten your table into two 1-D lists of equal length L:
 
        # 2) Grab degrees and caches in one go:
        
        deg =  self.summarized_graph.nodes[ntype].data[f"deg_{etype}"]
       
        deg1 = deg[node1s]  # shape (L,)
        deg2 = deg[node2s]            # shape (L,)

        
        su = cache[node1s]            # shape (L, D)
        sv = cache[node2s]
        cu = self.summarized_graph.nodes[ntype].data["node_size"][node1s].unsqueeze(1)
        cv = self.summarized_graph.nodes[ntype].data["node_size"][node2s].unsqueeze(1)
        cuv = cu + cv  # shape (L,)

        if self.multi_relations:
            return (su + sv ) / torch.sqrt((deg1 + deg2 + cuv.squeeze())).unsqueeze(1 )
        
                
        adj1 = self._get_adj(node1s, node2s, etype)
        adj2 = self._get_adj(node2s, node1s, etype)
        
        feat_u = self.summarized_graph.nodes[ntype].data["feat"][node1s]
        feat_v = self.summarized_graph.nodes[ntype].data["feat"][node2s]
        
        
        
            
        minus = torch.mul((adj2 / torch.sqrt(deg1 + cu.squeeze() )).unsqueeze(1), feat_u) + torch.mul((adj1 / torch.sqrt(deg2 + cv.squeeze() )).unsqueeze(1), feat_v) # + torch.matmul( (adj / torch.sqrt(deg2 + cv.squeeze() )), feat_v)
        
        

        # 3) Cluster‐size term (make sure cluster_sizes is a tensor):
      
        
        # 4) Single vectorized compute of h for all L pairs:
        #    (we broadcast / unsqueeze cuv into the right D-dimensional form)
        feat = (feat_u*cu + feat_v*cv) / (cu + cv)
        plus =     torch.mul((adj2 / torch.sqrt(deg1 + deg2 + cuv.squeeze() )).unsqueeze(1), feat) + torch.mul((adj1 / torch.sqrt(deg1 + deg2 + cuv.squeeze() )).unsqueeze(1), feat)
        if not self.feat_in_gcn:
            return  (su + sv - minus + plus)/ torch.sqrt((deg1 + deg2 + cuv.squeeze())).unsqueeze(1 )  #+   #)  # (L, D)

        h_all =  feat * ( cuv.squeeze() / ((deg1 + deg2 + cuv.squeeze()))).unsqueeze(1) + (su + sv - minus + plus)/ torch.sqrt((deg1 + deg2 + cuv.squeeze())).unsqueeze(1 )  #+   #)  # (L, D)
        
        return h_all
    
    def neigbor_approx_difference_per_pair(self,g, pairs,  d_node, c_node, infl_node, feat_node, etype):
        u, v = pairs[:,0], pairs[:,1]
        
        
        cu = c_node[u]
        cv = c_node[v]
        
        du = d_node[u]
        dv = d_node[v]
        
        iu = infl_node[u] 
        iv = infl_node[v]
        
        feat_u = feat_node[u]
        feat_v = feat_node[v]
        
        feat = (feat_u*cu.unsqueeze(1) + feat_v*cv.unsqueeze(1)) / (cu + cv).unsqueeze(1)
        
        
        neigh_cost_u = torch.norm( (feat / torch.sqrt(du + dv + cu + cv).unsqueeze(1)   - (feat_u / torch.sqrt(du + cu).unsqueeze(1))) , dim=1, p=self.norm_p)   * iu
        neigh_cost_v = torch.norm( (feat / torch.sqrt(du + dv + cu + cv).unsqueeze(1)   - (feat_v / torch.sqrt(dv + cv).unsqueeze(1))), dim=1, p=self.norm_p)   * iv
        #res = torch.where(mask,  neigh_cost_u + neigh_cost_v, torch.tensor(0.0, device=self.device))
        return neigh_cost_u + neigh_cost_v
    
    def neighbor_difference_per_pair(self,g, pairs, src_type, dst_type,  etype):
        """
        For every pair (u,v) and every neighbour n of u or v compute

            d_{(u,v),n} = [h_n + a_{un} h_u + a_{vn} h_v] - m_n

        Returns
        -------
        d            : (K, F_h)  one row per (pair, neighbour)
        pair_d_sum   : (P, F_h)  sum over neighbours for each pair
        """
        
        d_node_src = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"]
        c_node_src = self.summarized_graph.nodes[src_type].data["node_size"]
        feat_node_src = self.summarized_graph.nodes[src_type].data[f"feat"]
        if src_type == dst_type:
            h_node_dst = self.summarized_graph.nodes[dst_type].data[f"h{etype}"]
            d_node_dst = self.summarized_graph.nodes[dst_type].data[f"deg_{etype}"]
            c_node_dst  = self.summarized_graph.nodes[dst_type].data["node_size"]
            s_node_dst =  self.summarized_graph.nodes[dst_type].data[f"s{etype}"]
            feat_node_dst = self.summarized_graph.nodes[dst_type].data[f"feat"]
            
        else:
            h_node_dst = self.summarized_graph.nodes[dst_type].data[f"h{etype}"]
            d_node_dst = self.summarized_graph.nodes[dst_type].data[f"deg_{etype}"]
            c_node_dst  = self.summarized_graph.nodes[dst_type].data["node_size"]
            s_node_dst =  self.summarized_graph.nodes[dst_type].data[f"s{etype}"]
            feat_node_dst = self.summarized_graph.nodes[dst_type].data[f"feat"]
        
        
        a_edge = self.summarized_graph.edges[etype].data[f"adj"]
            
        
        #   
        
        start_time = time.time()   
        
        device = d_node_src.device
        pairs  = pairs.to(device)
        u, v   = pairs[:, 0], pairs[:, 1]          # (P,)
        P      = pairs.shape[0]

        # ------------------------------------------------------------------ #
        # 1 ▸ pick all edges touching any u or v                             #
        # ------------------------------------------------------------------ #
        if src_type == dst_type:
            src, dst, _ = g.edges(form='all', etype=etype)
        else:
            src, dst, _ = g.edges(form='all', etype=etype)
        src, dst    = src.to(device), dst.to(device)
       # print("get edges",  time.time()  - start_time )
        start_time = time.time()   
        
        touches = (
         (dst[None, :] == u[:, None]) |      # dst is u
                  (dst[None, :] == v[:, None])        # dst is v
            )
      #  print("create touch",  time.time()  - start_time )
        
        start_time = time.time()   
        
        pair_idx, edge_idx = touches.nonzero(as_tuple=True)      # (K,)

        e_src, e_dst      = src[edge_idx], dst[edge_idx]
        a_val             = a_edge[edge_idx].view(-1)            # (K,)

        nbr               =  src[edge_idx]       #torch.where(is_src_endpoint, e_dst, e_src)

        connects_u        = (dst[edge_idx] == u[pair_idx])
        connects_v        =  (dst[edge_idx] == v[pair_idx])                                
     #   print("create connection",  time.time()  - start_time )
        # ------------------------------------------------------------------ #
        # 2 ▸ heterograph: pair ─knows─> neighbour                          #
        # ------------------------------------------------------------------ #
     #   t = torch.stack((pair_idx)
        hg = dgl.heterograph(
            {('pair', 'knows', 'node'): (pair_idx, nbr)},
            
            num_nodes_dict={'pair': P, 'node': g.num_nodes(ntype=dst_type)}
        )

        hg.nodes['pair'].data['id_u'] = u
        hg.nodes["pair"].data["id_v"] = v
        hg.nodes['pair'].data['f_u'] = feat_node_src[u]
        hg.nodes['pair'].data['f_v'] = feat_node_src[v]
        
        hg.nodes['pair'].data['d_u'] = d_node_src[u]
        hg.nodes['pair'].data['d_v'] = d_node_src[v]
        
        hg.nodes['pair'].data['c_u'] = c_node_src[u]
        hg.nodes['pair'].data['c_v'] = c_node_src[v]
        
        
        
        hg.nodes["node"].data["id_n"] = torch.arange(d_node_dst.shape[0], device=self.device)
        hg.nodes['node'].data['d_n'] = d_node_dst
        hg.nodes['node'].data['c_n'] = c_node_dst
        hg.nodes['node'].data['s_n'] = s_node_dst
        hg.nodes['node'].data['f_n'] = feat_node_dst
        hg.nodes['node'].data['h_n'] = h_node_dst


        hg.edges['knows'].data['a_un'] = (a_val) * connects_u.float()
        hg.edges['knows'].data['a_vn'] = (a_val )* connects_v.float()
        hg.edges['knows'].data['a_uvn'] = (a_val) * (connects_u.float() +connects_v.float())
    #    hg.edges['knows'].data[''] = (a_val )* connects_v.float()
        start_time = time.time()
        hg = hg.to("cpu").to_simple(hg,  aggregator="sum", copy_edata=True)
        hg = hg.to(self.device)
      #  print("to simple", time.time() -start_time)
        
        # ------------------------------------------------------------------ #
        # 3 ▸ fused CUDA kernel per (pair,nbr) edge                          #
        # ------------------------------------------------------------------ #
        #mask = torch.logical_and(hg.nodes['pair'].data["id_u"] == 2, hg.nodes['pair'].data["id_v"] == 4)
        def edge_formula(edges):
            ##mask = torch.logical_and(edges.src["id_u"] == 2, edges.src["id_v"] == 4)
            if src_type == dst_type:
                mask =  torch.logical_or(edges.dst["id_n"] == edges.src['id_u'], edges.dst["id_n"] == edges.src['id_v'])
                mask = mask == False
            if self.feat_in_gcn:
                h_prime = (edges.dst["c_n"] / (edges.dst["d_n"] + edges.dst["c_n"])).unsqueeze(1) * edges.dst["f_n"]
            
                h_prime += (1 / torch.sqrt(edges.dst["d_n"] + edges.dst["c_n"])).unsqueeze(1) * edges.dst["s_n"]
            else: 
                h_prime = (1 / torch.sqrt(edges.dst["d_n"] + edges.dst["c_n"])).unsqueeze(1) * edges.dst["s_n"]
                
            feat_uv = (edges.src['f_u'] * edges.src['c_u'].unsqueeze(1) + edges.src['f_v'] * edges.src['c_v'].unsqueeze(1)) / (edges.src['c_u'] + edges.src['c_v']).unsqueeze(1)
            num = (edges.data['a_uvn'].unsqueeze(-1) ) *  feat_uv
            denom =  torch.sqrt( (edges.dst["d_n"] + edges.dst["c_n"] ) * (edges.src["d_u"] + edges.src["c_u"]  +edges.src["d_v"] + edges.src["c_v"] ) ) 
            h_prime += num  / denom.unsqueeze(1)
            
            
            num = (edges.data['a_un'].unsqueeze(-1) ) *  edges.src['f_u']
            denom =  torch.sqrt( (edges.dst["d_n"] + edges.dst["c_n"] ) * (edges.src["d_u"] + edges.src["c_u"] ) ) 
            h_prime -= num  / denom.unsqueeze(1)
            
            num = (edges.data['a_vn'].unsqueeze(-1) ) *  edges.src['f_v']
            denom =  torch.sqrt( (edges.dst["d_n"] + edges.dst["c_n"] ) * (edges.src["d_v"] + edges.src["c_v"] ) ) 
            h_prime -= num  / denom.unsqueeze(1)
            
            d = torch.norm(h_prime - edges.dst['h_n'], p=self.norm_p, dim=1)
            if src_type == dst_type:
                d = torch.where(mask, d, torch.tensor(0,device=self.device))
            return {'d': d}
        start_time = time.time()
        hg.apply_edges(edge_formula, etype=('pair', 'knows', 'node'))

        d = hg.edges['knows'].data['d']                   # (K, F_h)

        # ------------------------------------------------------------------ #
        # 4 ▸ aggregate d over neighbours for each pair                      #
        # ------------------------------------------------------------------ #
        pair_idx,_ = hg.edges(etype="knows")
        pair_d_sum = scatter_add(d, pair_idx, dim=0, dim_size=P) 
        #print("edges & scatter", time.time() -start_time)
        # (P, F_h)

        return pair_d_sum

    
    
    def _create_neighbor_costs(self):
        start_time = time.time()
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            if dst_type not in self.merge_graphs:
                continue 
            merge_graph = self.merge_graphs[dst_type] 
            
            merge_node_u, merge_node_v, merge_graph_eid = merge_graph.edges(form="all")
            pairs = torch.stack((merge_node_u, merge_node_v) , dim=1)
                
            if self.approx_neigh:
                merge_graph = self.merge_graphs[src_type] 
            
                merge_node_u, merge_node_v, merge_graph_eid = merge_graph.edges(form="all")
                pairs = torch.stack((merge_node_u, merge_node_v) , dim=1)
            
                h_node = self.summarized_graph.nodes[src_type].data[f"h{etype}"]
                d_node = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"]
                c_node = cv = self.summarized_graph.nodes[src_type].data["node_size"]
                s_node =  self.summarized_graph.nodes[src_type].data[f"s{etype}"]
                feat_node = self.summarized_graph.nodes[src_type].data[f"feat"]
                
                
                
                h_node = self.summarized_graph.nodes[src_type].data[f"h{etype}"]
                d_node = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"]
                c_node = cv = self.summarized_graph.nodes[src_type].data["node_size"]
                s_node =  self.summarized_graph.nodes[src_type].data[f"s{etype}"]
                feat_node = self.summarized_graph.nodes[src_type].data[f"feat"]
                infl_node = self.summarized_graph.nodes[src_type].data[f"i{etype}"]
          
                a_edge = self.summarized_graph.edges[etype].data[f"adj"]
           
            #   
                neighbors_cost  = self.neigbor_approx_difference_per_pair(self.summarized_graph, pairs,d_node, c_node, infl_node, feat_node, etype)
            else:            
                neighbors_cost  = self.neighbor_difference_per_pair(self.summarized_graph, pairs,dst_type, src_type, etype)
            merge_graph.edata[f"costs_neig_{etype}"] = neighbors_cost
      #  print("_create_neighbor_costs", time.time() - start_time)
    
    
    def _h_cca_costs(self):
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
        
            if src_type not in self.merge_graphs:    
                continue
            
            
            src, dst = self.merge_graphs[src_type].edges()
             
            # compute all merged h representations in one go
            H_merged = self._create_h_merged(src,dst, src_type, etype)
    
            # gather representations
            hu = self.summarized_graph.nodes[src_type].data[f"h{etype}"][src]  # [P, H]
            hv = self.summarized_graph.nodes[src_type].data[f"h{etype}"][dst]  # [P, H]
            # build a dense [num_src, hidden] tensor
            #H_tensor =  torch.tensor([v for k,v in  H_merged.items()] , device=device)
            huv = H_merged                               # [P, H]
            feat_u = self.summarized_graph.nodes[src_type].data["feat"][src]
            feat_v = self.summarized_graph.nodes[src_type].data["feat"][dst]
        
            cu = self.summarized_graph.nodes[src_type].data["node_size"][src]

            cv = self.summarized_graph.nodes[src_type].data["node_size"][dst]
            du = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"][src]
            dv = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"][dst]
            feat = (feat_u*cu.unsqueeze(1) + feat_v*cv.unsqueeze(1)) / (cu + cv ).unsqueeze(1)
            
            feat_u,hu_cca = self.ccas[etype].transform(feat_u, hu)
            feat_v,hv_cca = self.ccas[etype].transform(feat_v, hv)
            feat_uv, huv_cca = self.ccas[etype].transform(feat, huv)
            
            # feat_u,hu_cca = torch.from_numpy(feat_u).to(self.device), torch.from_numpy(hu_cca).to(self.device)
            # feat_v,hv_cca = torch.from_numpy(feat_v).to(self.device), torch.from_numpy(hv_cca).to(self.device)
            # feat_uv, huv_cca = torch.from_numpy(feat_uv).to(self.device), torch.from_numpy(huv_cca).to(self.device)
          #  print("du hurensohn!!")
            
            feat_uv =  feat_uv* (cu + cv ).unsqueeze(1) / (du +dv + cv+ cu ).unsqueeze(1) 
            feat_u = (feat_u * cu.unsqueeze(1)) / (du + cu).unsqueeze(1)
            feat_v = (feat_v * cv.unsqueeze(1)) / (dv + cv).unsqueeze(1)
            self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"] = torch.norm(feat_uv - feat_u + huv_cca  - hu_cca, p=self.norm_p, dim=1)
            self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"] =  torch.norm(feat_uv - feat_v + huv_cca  - hv_cca, p=self.norm_p, dim=1)
       # pr
            
        
       
        
    def _h_costs(self):
        start_time = time.time()
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            # ensure nested dict
            
            if src_type not in self.merge_graphs:    
                continue
            
            
            src, dst = self.merge_graphs[src_type].edges()
            
             
            # compute all merged h representations in one go
            H_merged = self._create_h_merged(src,dst, src_type, etype)
            
            
            # flatten all (u,v) pairs same as above
            

            node1_ids = src  # [P]
            node2_ids = dst    # [P]

            # gather representations
            h1 = self.summarized_graph.nodes[src_type].data[f"h{etype}"][node1_ids]  # [P, H]
            h2 = self.summarized_graph.nodes[src_type].data[f"h{etype}"][node2_ids]  # [P, H]
            # build a dense [num_src, hidden] tensor
            #H_tensor =  torch.tensor([v for k,v in  H_merged.items()] , device=device)
            merged = H_merged                               # [P, H]
            
            
            
            
            
            self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"] = torch.norm(merged - h1, p=self.norm_p, dim=1)
            self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"] =  torch.norm(merged - h2, p=self.norm_p, dim=1)
       # print("_h_costs", time.time() - start_time)

        
        
    def _inner_product(self):
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            # ensure nested dict
                
                
            
            
            src, dst = self.merge_graphs[src_type].edges()
            
             
            # compute all merged h representations in one go
            H_merged = self._create_h_merged(src,dst, src_type, etype)
            
            
            # flatten all (u,v) pairs same as above
            

            node1_ids = src  # [P]
            node2_ids = dst    # [P]

            # gather representations
            h1 = self.summarized_graph.nodes[src_type].data[f"h{etype}"][node1_ids]  # [P, H]
            h2 = self.summarized_graph.nodes[src_type].data[f"h{etype}"][node2_ids]  # [P, H]
            # build a dense [num_src, hidden] tensor
            #H_tensor =  torch.tensor([v for k,v in  H_merged.items()] , device=device)
            merged = H_merged                               # [P, H]
            
            feat_u = self.summarized_graph.nodes[src_type].data["feat"][node1_ids]
            feat_v = self.summarized_graph.nodes[src_type].data["feat"][node2_ids]
        
            cu = self.summarized_graph.nodes[src_type].data["node_size"][node1_ids]

            cv = self.summarized_graph.nodes[src_type].data["node_size"][node2_ids]
            
            du = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"][node1_ids]
            dv = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"][node2_ids]

            feat = (feat_u*cu.unsqueeze(1) + feat_v*cv.unsqueeze(1)) / (cu + cv +du + dv).unsqueeze(1)
            feat_u = (feat_u * cu.unsqueeze(1)) / (du + cu).unsqueeze(1)
            feat_v = (feat_v * cv.unsqueeze(1)) / (dv + cv).unsqueeze(1)

            
            # L1 cos
            
            self.merge_graphs[src_type].edata[f"costs_inner_{etype}_u"] = torch.norm(feat - feat_u - merged  + h1, p=self.norm_p, dim=1)
            self.merge_graphs[src_type].edata[f"costs_inner_{etype}_v"] = torch.norm(feat - feat_v - merged + h2, p=self.norm_p, dim=1)
                    
         
    def _self_feature_costs(self, ntype):
        src, dst = self.merge_graphs[ntype].edges()
        node1_ids = src  # [P]
        node2_ids = dst  
        feat_u = self.summarized_graph.nodes[ntype].data["feat"][node1_ids]
        feat_v = self.summarized_graph.nodes[ntype].data["feat"][node2_ids]
        
        cu = self.summarized_graph.nodes[ntype].data["node_size"][node1_ids]

        cv = self.summarized_graph.nodes[ntype].data["node_size"][node2_ids]
        self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_u"] = torch.zeros(node1_ids.shape[0], device=self.device)
        self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_v"] = torch.zeros(node2_ids.shape[0], device=self.device)
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            if ntype != src_type:
                continue
            du = self.summarized_graph.nodes[ntype].data[f"deg_{etype}"][node1_ids]
            dv = self.summarized_graph.nodes[ntype].data[f"deg_{etype}"][node2_ids]

            feat = (feat_u*cu.unsqueeze(1) + feat_v*cv.unsqueeze(1)) / (cu + cv +du + dv).unsqueeze(1)
            feat_u = (feat_u * cu.unsqueeze(1)) / (du + cu).unsqueeze(1)
            feat_v = (feat_v * cv.unsqueeze(1)) / (dv + cv).unsqueeze(1)
        
        
            self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_u"] += torch.norm(feat - feat_u, p=self.norm_p, dim=1)
            self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_v"] += torch.norm(feat - feat_v, p=self.norm_p, dim=1)
    
    def _sum_costs_feat_in_rgc(self):
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            
            self.merge_graphs[src_type].edata["costs_u"] =  self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"]
            
            
            self.merge_graphs[src_type].edata["costs_v"] =  self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"] 
            

        
            
            
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            self.merge_graphs[src_type].edata["costs"] = self.merge_graphs[src_type].edata["costs_u"]
            self.merge_graphs[src_type].edata["costs"] += self.merge_graphs[src_type].edata["costs_v"]
            
        
            self.merge_graphs[src_type].edata["costs"] += self.merge_graphs[src_type].edata[f"costs_neig_{etype}"]
            
    
    def _sum_costs_feat_sep_with_inner(self):
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            
            self.merge_graphs[src_type].edata["costs_u"] = 2* torch.pow(self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"],2) 
            self.merge_graphs[src_type].edata["costs_u"] -= torch.pow(self.merge_graphs[src_type].edata[f"costs_inner_{etype}_u"], 2)  
            
            
            self.merge_graphs[src_type].edata["costs_v"] = 2* torch.pow(self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"],2) 
            self.merge_graphs[src_type].edata["costs_v"] -= torch.pow(self.merge_graphs[src_type].edata[f"costs_inner_{etype}_v"], 2)  
            
            
        for ntype in self.summarized_graph.ntypes:
            self._self_feature_costs(ntype)
            self.merge_graphs[ntype].edata["costs_u"] += 2* torch.pow(self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_u"] , 2)
            self.merge_graphs[ntype].edata["costs_v"] += 2* torch.pow(self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_v"] , 2)
        
            
            
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            self.merge_graphs[src_type].edata["costs"] = torch.sqrt(self.merge_graphs[src_type].edata["costs_u"])
            self.merge_graphs[src_type].edata["costs"] += torch.sqrt(self.merge_graphs[src_type].edata["costs_v"])
            
        
            self.merge_graphs[src_type].edata["costs"] += self.merge_graphs[src_type].edata[f"costs_neig_{etype}"]
    
            
    
    
    def _sum_costs_feat_sep(self):
        for ntype in self.summarized_graph.ntypes:
            self.merge_graphs[ntype].edata["costs_u"] = torch.zeros(self.merge_graphs[ntype].num_edges(), device=self.device)
            self.merge_graphs[ntype].edata["costs_v"] = torch.zeros(self.merge_graphs[ntype].num_edges(), device=self.device)
        
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
           # print("Hi")
            if src_type not in self.merge_graphs:
                continue
        
            self.merge_graphs[src_type].edata["costs_u"] += self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"] 
            
            
            self.merge_graphs[src_type].edata["costs_v"] += self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"]
            
            
        for ntype in self.summarized_graph.ntypes:
            if ntype not in self.merge_graphs:
                continue
            continue
            self._self_feature_costs(ntype)
            self.merge_graphs[ntype].edata["costs_u"] +=  self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_u"] 
            self.merge_graphs[ntype].edata["costs_v"] +=  self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_v"] 
        
            
            
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            if src_type not in self.merge_graphs:
                continue
            
            self.merge_graphs[dst_type].edata["costs"] =  self.merge_graphs[dst_type].edata["costs_u"]
            self.merge_graphs[dst_type].edata["costs"] += self.merge_graphs[dst_type].edata["costs_v"]
            
            if f"costs_neig_{etype}" in self.merge_graphs[dst_type].edata:
                self.merge_graphs[dst_type].edata["costs"] += self.merge_graphs[dst_type].edata[f"costs_neig_{etype}"]
    
    
    def _create_costs(self):
        if self.use_cca:
            self._h_cca_costs()
        else:
            self._h_costs()
        self._create_neighbor_costs()
        if self.feat_in_gcn:
            self._sum_costs_feat_in_rgc()
        else:
            if self.inner_product:
                self._inner_product()
                self._sum_costs_feat_sep_with_inner()
            else:
                self._sum_costs_feat_sep()
           
    def _init_merge_graphs(self, type_pairs):
        self.merge_graphs = dict()
        for ntype, pairs in type_pairs.items():
            self.merge_graphs[ntype] = dgl.graph(([], []), num_nodes=self.summarized_graph.number_of_nodes(ntype=ntype), device=self.device)
            self.merge_graphs[ntype].add_edges(pairs[:,0],pairs[:,1])
        self._create_costs()
        
        
    def _update_merge_graph(self, mappings):
        self.reduce_merge_graph(mappings)
        self._create_costs()
        self.candidates = self._find_lowest_costs()
        
        
    
        
      
         

