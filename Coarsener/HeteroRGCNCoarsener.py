
import torch
import dgl
from Coarsener.HeteroCoarsener import HeteroCoarsener
import time


class HeteroRGCNCoarsener(HeteroCoarsener):
    
    
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
            deg_out = self.summarized_graph.out_degrees(etype= etype)
            n_src =  self.summarized_graph.num_nodes(src_type)
            c = torch.ones((n_src), device=self.device)
            self.summarized_graph.nodes[src_type].data['node_size']  = c
            
            inv_sqrt_out = torch.rsqrt(deg_out + self.summarized_graph.nodes[src_type].data['node_size'])
            

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
                
            
            s_e = feat_v * inv_sqrt_out[v].unsqueeze(-1)       # [E, D]
            # Scatter-add to compute S at source nodes
           
            S_tensor = torch.zeros((n_src, feat_dim), device=self.device)
            S_tensor = S_tensor.index_add(0, u, s_e)
            infl = torch.zeros(n_src, device=self.device)
            infl = infl.index_add(0, u, inv_sqrt_out[v])

            # Compute H = D_out^{-1/2} * S
            H_tensor = inv_sqrt_out.unsqueeze(-1) * S_tensor

            # Store in coarsened_graph
            self.summarized_graph.nodes[src_type].data[f'i{etype}'] = infl
            
            self.summarized_graph.nodes[src_type].data[f's{etype}'] = S_tensor
           
            self.summarized_graph.nodes[src_type].data[f'h{etype}'] = H_tensor + ((feats / (deg_out + c).unsqueeze(1) ))
            

        print("_create_h_spatial_rgcn", time.time() - start_time)


