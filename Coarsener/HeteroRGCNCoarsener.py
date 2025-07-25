
import torch
import dgl
from Coarsener.HeteroCoarsener import HeteroCoarsener
import time
import dgl.backend as F         # this is the backend (torch, TF, etc.)
from torch_scatter import scatter_add   
import numpy as np

import torch
    
import torch

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from Projections.NonLinStochCCA import NonlinearStochasticCCA
from Projections.JLRandom import JLRandomProjection




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
            if True or self.device == "cpu":
                A = g.adj_external(etype=etype).cpu().to_dense().to(self.device)
            else:
                A = g.adj(etype=etype)
                
            D = torch.diag(torch.rsqrt(torch.sum(A, dim=1)))
            feat =  self.summarized_graph.nodes[src_type].data['feat_pca']
            H =  torch.pow((D @A @  D), k ) @ feat
            self.summarized_graph.nodes[src_type].data[f"SGC{etype}"] = H
    
    
    def _apply_random_projection_on_features(self):
        for ntype in self.summarized_graph.ntypes:
            feats = self.summarized_graph.nodes[ntype].data['feat'].to(self.device)
            random_projection =self.projection_cls(feats.shape[1], 50, device=self.device)
            self.summarized_graph.nodes[ntype].data['feat'] = random_projection.forward(feats)
                
        
        pass
    
    
        
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
            deg_out = self.original_graph.out_degrees(etype= etype)
            n_src =  self.summarized_graph.num_nodes(src_type)
            n_dst =  self.summarized_graph.num_nodes(dst_type)
            c = self.summarized_graph.nodes[src_type].data['node_size']
            
            inv_sqrt_out = torch.rsqrt(deg_out + self.summarized_graph.nodes[src_type].data['node_size'])
            inv_sqrt_in = torch.rsqrt(self.summarized_graph.in_degrees(etype= etype) + self.summarized_graph.nodes[dst_type].data['node_size'])

            # Load features or use scalar 1
            if has_feat:
                feats = self.summarized_graph.nodes[dst_type].data['feat_pca'].to(self.device)
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
                if self.multi_relations:
                    feats = self.summarized_graph.nodes[src_type].data['feat_pca'].to(self.device)
                    self.summarized_graph.nodes[src_type].data[f'h{etype}'] = H_tensor 
                else:
                    self.summarized_graph.nodes[src_type].data[f'h{etype}'] = H_tensor + ((feats / (deg_out + c).unsqueeze(1) ))
            else:
                self.summarized_graph.nodes[src_type].data[f'h{etype}'] = H_tensor
           
           
        if self.cca_cls:
            self.ccas = dict()
            for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
                feat_src =   self.summarized_graph.nodes[src_type].data['feat_pca']
                h_src = self.summarized_graph.nodes[src_type].data[f'h{etype}']
                
                cca = self.cca_cls(feat_src.shape[1], h_src.shape[1], n_components=h_src.shape[1], device=self.device)
                cca.fit(feat_src, h_src) 
                self.ccas[etype] = cca
        # else:
        #     for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
        #         self.summarized_graph.nodes[src_type].data[f'h{etype}'] = 
     #   print("_create__spatial_rgcn", time.time() - start_time)

    
    def _create_h_merged(self,  node1s, node2s,ntype, etype):
        cache = self.summarized_graph.nodes[ntype].data[f's{etype}']
        
        # 1) Flatten your table into two 1-D lists of equal length L:
 
        # 2) Grab degrees and caches in one go:
        
        deg =  self.summarized_graph.nodes[ntype].data[f"deg_{etype}"]
       
        deg1 = deg[node1s]  # shape (L,)
        deg2 = deg[node2s]            # shape (L,)

        if self.ccas:
            cache = self.ccas[etype].transform(X2=cache)
        su = cache[node1s]            # shape (L, D)
        sv = cache[node2s]
        cu = self.summarized_graph.nodes[ntype].data["node_size"][node1s].unsqueeze(1)
        cv = self.summarized_graph.nodes[ntype].data["node_size"][node2s].unsqueeze(1)
        cuv = cu + cv  # shape (L,)

        if self.multi_relations:
            return (su + sv ) / torch.sqrt((deg1 + deg2 + cuv.squeeze())).unsqueeze(1 )
        
                
        adj1 = self._get_adj(node1s, node2s, etype)
        adj2 = self._get_adj(node2s, node1s, etype)
        
        feat_u = self.summarized_graph.nodes[ntype].data["feat_pca"][node1s]
        feat_v = self.summarized_graph.nodes[ntype].data["feat_pca"][node2s]
        
        
        
            
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
        if self.cca_cls:
            if self.multi_relations:
                dst_type = etype.split("to")[1]
                src_type = etype.split("to")[0]
                etype_around = f"{dst_type}to{src_type}"
            else:
                etype_around = etype
            feat_node = self.ccas[etype_around].transform(X1=feat_node)
        feat_u = feat_node[u]
        feat_v = feat_node[v]
        
        feat = (feat_u*cu.unsqueeze(1) + feat_v*cv.unsqueeze(1)) / (cu + cv).unsqueeze(1)
        # if self.cca_cls:
        #     if self.multi_relations:
        #         dst_type = etype.split("to")[1]
        #         src_type = etype.split("to")[0]
        #         etype_around = f"{dst_type}to{src_type}"
        #     else:
        #         etype_around = etype
        #     feat_u =  self.ccas[etype_around].transform(feat_u)
        #     feat_v = self.ccas[etype_around].transform(feat_v)
        #     feat =  self.ccas[etype_around].transform(feat)
        if self.use_cos_sim:
            neigh_cost_u = F.cosine_similarity( feat / torch.sqrt(du + dv + cu + cv).unsqueeze(1)  , (feat_u / torch.sqrt(du + cu).unsqueeze(1)) , dim=1)   * iu
            neigh_cost_v = F.cosine_similarity( feat / torch.sqrt(du + dv + cu + cv).unsqueeze(1)   , (feat_v / torch.sqrt(dv + cv).unsqueeze(1)), dim=1)   * iv
        
        else:
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
        feat_node_src = self.summarized_graph.nodes[src_type].data[f"feat_pca"]
        if src_type == dst_type:
            h_node_dst = self.summarized_graph.nodes[dst_type].data[f"h{etype}"]
            d_node_dst = self.summarized_graph.nodes[dst_type].data[f"deg_{etype}"]
            c_node_dst  = self.summarized_graph.nodes[dst_type].data["node_size"]
            s_node_dst =  self.summarized_graph.nodes[dst_type].data[f"s{etype}"]
            feat_node_dst = self.summarized_graph.nodes[dst_type].data[f"feat_pca"]
            
        else:
            h_node_dst = self.summarized_graph.nodes[dst_type].data[f"h{etype}"]
            d_node_dst = self.summarized_graph.nodes[dst_type].data[f"deg_{etype}"]
            c_node_dst  = self.summarized_graph.nodes[dst_type].data["node_size"]
            s_node_dst =  self.summarized_graph.nodes[dst_type].data[f"s{etype}"]
            feat_node_dst = self.summarized_graph.nodes[dst_type].data[f"feat_pca"]
        
        
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
            if self.use_cos_sim:
                d = F.cosine_similarity(h_prime, edges.dst['h_n'], dim=1)
            
            else:
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
                
                merge_node_u, merge_node_v, merge_graph_eid = merge_graph.edges(form="all")
                pairs = torch.stack((merge_node_u, merge_node_v) , dim=1)
            
               # h_node = self.summarized_graph.nodes[dst_type].data[f"h{etype}"]
                d_node = self.summarized_graph.nodes[dst_type].data[f"deg_{etype}"]
                c_node = cv = self.summarized_graph.nodes[dst_type].data["node_size"]
                #s_node =  self.summarized_graph.nodes[dst_type].data[f"s{etype}"]
                feat_node = self.summarized_graph.nodes[dst_type].data[f"feat_pca"]
                
                
                
                #h_node = self.summarized_graph.nodes[dst_type].data[f"h{etype}"]
                d_node = self.summarized_graph.nodes[dst_type].data[f"deg_{etype}"]
                c_node = cv = self.summarized_graph.nodes[dst_type].data["node_size"]
                #s_node =  self.summarized_graph.nodes[dst_type].data[f"s{etype}"]
                feat_node = self.summarized_graph.nodes[dst_type].data[f"feat_pca"]
                if self.multi_relations:
                    etype_around = f"{dst_type}to{src_type}"
                else:
                    etype_around = etype
                
                infl_node = self.summarized_graph.nodes[dst_type].data[f"i{etype_around}"]
          
           #     a_edge = self.summarized_graph.edges[etype].data[f"adj"]
           
            #   
                if self.batch_size:
                    costs = []
                
                    for i in range(0, len(pairs[:,0]), self.batch_size):
                        s, d = pairs[:,0][i:i+self.batch_size], pairs[:,1][i:i+self.batch_size]
                        pairs_batch = torch.stack((s,d), dim=1)
                        costs.append(self.neigbor_approx_difference_per_pair(self.summarized_graph, pairs_batch,d_node, c_node, infl_node, feat_node, etype))
                    neighbors_cost  =torch.cat(costs, dim=0)
                else: 
                    neighbors_cost = self.neigbor_approx_difference_per_pair(self.summarized_graph, pairs,d_node, c_node, infl_node, feat_node, etype )
            else:            
                neighbors_cost  = self.neighbor_difference_per_pair(self.summarized_graph, pairs,dst_type, src_type, etype)
            merge_graph.edata[f"costs_neig_{etype}"] = neighbors_cost
      #  print("_create_neighbor_costs", time.time() - start_time)
    
    def _h_cca_costs_batched(self):
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
        
            if src_type not in self.merge_graphs:    
                continue
            
            
            src_all, dst_all = self.merge_graphs[src_type].edges()
            costs_u = []
            costs_v = []
           
            for i in range(0, len(src_all), self.batch_size):
                src = src_all[i:i+self.batch_size]
                dst = dst_all[i:i+self.batch_size]
 
                # compute all merged h representations in one go
                H_merged = self._create_h_merged(src,dst, src_type, etype)

                H =  self.summarized_graph.nodes[src_type].data[f"h{etype}"]
                H_cca = self.ccas[etype].transform(X2 =H)
                
                # gather representations
                hu_cca = H_cca[src]  # [P, H]
                hv_cca = H_cca[dst]  # [P, H]
                # build a dense [num_src, hidden] tensor
                #H_tensor =  torch.tensor([v for k,v in  H_merged.items()] , device=device)
                huv_cca = H_merged      
                feat_cca =   self.ccas[etype].transform(X1 =self.summarized_graph.nodes[src_type].data["feat_pca"])                       # [P, H]
                feat_u = feat_cca[src]
                feat_v = feat_cca[dst]
            
                cu = self.summarized_graph.nodes[src_type].data["node_size"][src]

                cv = self.summarized_graph.nodes[src_type].data["node_size"][dst]
                du = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"][src]
                dv = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"][dst]
                # feat_u= self.ccas[etype].transform(X1 =feat_u)
                # feat_v= self.ccas[etype].transform(X1 =feat_v)
                # feat_uv = self.ccas[etype].transform(X1 =feat)
                # # feat_u,hu_cca = torch.from_numpy(feat_u).to(self.device), torch.from_numpy(hu_cca).to(self.device)
                # feat_v,hv_cca = torch.from_numpy(feat_v).to(self.device), torch.from_numpy(hv_cca).to(self.device)
                # feat_uv, huv_cca = torch.from_numpy(feat_uv).to(self.device), torch.from_numpy(huv_cca).to(self.device)
                #  print("du hurensohn!!")
                feat_uv = (feat_u*cu.unsqueeze(1) + feat_v*cv.unsqueeze(1)) / (cu + cv ).unsqueeze(1)
                feat_uv =  feat_uv* (cu + cv ).unsqueeze(1) / (du +dv + cv+ cu ).unsqueeze(1) 
                feat_u = (feat_u * cu.unsqueeze(1)) / (du + cu).unsqueeze(1)
                feat_v = (feat_v * cv.unsqueeze(1)) / (dv + cv).unsqueeze(1)
                
                
                if self.use_cos_sim:
                    costs_u.append( F.cosine_similarity(feat_uv - feat_u , huv_cca  - hu_cca,  dim=1))
                    costs_v.append(F.cosine_similarity(feat_uv - feat_v , huv_cca  - hv_cca,  dim=1))
                else:
                    costs_u.append( torch.norm(feat_uv - feat_u + huv_cca  - hu_cca, p=self.norm_p, dim=1))
                    costs_v.append(  torch.norm(feat_uv - feat_v + huv_cca  - hv_cca, p=self.norm_p, dim=1))
    

            self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"]  = torch.cat(costs_u, dim=0)
            self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"] = torch.cat(costs_v, dim=0)
    
    def _h_cca_costs(self):
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
        
            if src_type not in self.merge_graphs:    
                continue
            
            
            src, dst = self.merge_graphs[src_type].edges()
             
            # compute all merged h representations in one go
            H_merged = self._create_h_merged(src,dst, src_type, etype)
            H =  self.summarized_graph.nodes[src_type].data[f"h{etype}"]
            H_cca = self.ccas[etype].transform(X2 =H)
            # gather representations
            hu_cca = H_cca[src]  # [P, H]
            hv_cca = H_cca[dst]  # [P, H]
            # build a dense [num_src, hidden] tensor
            #H_tensor =  torch.tensor([v for k,v in  H_merged.items()] , device=device)
            huv_cca = H_merged                               # [P, H]
            feat_cca =   self.ccas[etype].transform(X1 =self.summarized_graph.nodes[src_type].data["feat_pca"])  
            feat_u = feat_cca[src]
            feat_v = feat_cca[dst]
        
            cu = self.summarized_graph.nodes[src_type].data["node_size"][src]

            cv = self.summarized_graph.nodes[src_type].data["node_size"][dst]
            du = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"][src]
            dv = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"][dst]
            feat_uv = (feat_u*cu.unsqueeze(1) + feat_v*cv.unsqueeze(1)) / (cu + cv ).unsqueeze(1)
          #  print(src_type, etype)
            # feat_u,hu_cca = self.ccas[etype].transform(feat_u, hu)
            # feat_v,hv_cca = self.ccas[etype].transform(feat_v, hv)
            # feat_uv, huv_cca = self.ccas[etype].transform(feat, huv)
            
         
            
            feat_uv =  feat_uv* (cu + cv ).unsqueeze(1) / (du +dv + cv+ cu ).unsqueeze(1) 
            feat_u = (feat_u * cu.unsqueeze(1)) / (du + cu).unsqueeze(1)
            feat_v = (feat_v * cv.unsqueeze(1)) / (dv + cv).unsqueeze(1)
            if self.use_cos_sim:
                self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"] = F.cosine_similarity(feat_uv - feat_u , huv_cca  - hu_cca,  dim=1)
                self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"] =  F.cosine_similarity(feat_uv - feat_v , huv_cca  - hv_cca,  dim=1)
       # pr
            else:
                self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"] = torch.norm(feat_uv - feat_u + huv_cca  - hu_cca, p=self.norm_p, dim=1) 
                self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"] =  torch.norm(feat_uv - feat_v + huv_cca  - hv_cca, p=self.norm_p, dim=1) 
       # pr
            
        
       
    def _h_costs_batched(self):
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
       
            if src_type not in self.merge_graphs:
                continue

            src_all, dst_all = self.merge_graphs[src_type].edges()

            
            costs_u = []
            costs_v = []
           
            for i in range(0, len(src_all), self.batch_size):
                src = src_all[i:i+self.batch_size]
                dst = dst_all[i:i+self.batch_size]

                H_merged = self._create_h_merged(src, dst, src_type, etype)  # [B, H]

                h1 = self.summarized_graph.nodes[src_type].data[f"h{etype}"][src]  # [B, H]
                h2 = self.summarized_graph.nodes[src_type].data[f"h{etype}"][dst]  # [B, H]

                if self.use_cos_sim:
                    cost_u = F.cosine_similarity(H_merged, h1, dim=1)  # [B]
                    cost_v = F.cosine_similarity(H_merged, h2, dim=1)  # [B]
                else:
                    cost_u = torch.norm(H_merged - h1, p=self.norm_p, dim=1)  # [B]
                    cost_v = torch.norm(H_merged - h2, p=self.norm_p, dim=1)  # [B]

                costs_u.append(cost_u)
                costs_v.append(cost_v)

            # Concatenate batched results
            self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"] = torch.cat(costs_u, dim=0)
            self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"] = torch.cat(costs_v, dim=0)
        
    
    
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
            
            
            
            if self.use_cos_sim:
                self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"] = F.cosine_similarity(merged, h1, dim=1)
                self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"] =  F.cosine_similarity(merged, h2, dim=1)
            else:
            
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
            
            feat_u = self.summarized_graph.nodes[src_type].data["feat_pca"][node1_ids]
            feat_v = self.summarized_graph.nodes[src_type].data["feat_pca"][node2_ids]
        
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
                    
         
    def _feature_costs(self):
        for ntype in self.summarized_graph.ntypes:
            if not ntype in self.merge_graphs.keys():
                continue
            src, dst = self.merge_graphs[ntype].edges()
            node1_ids = src  # [P]
            node2_ids = dst  
            feat_u = self.summarized_graph.nodes[ntype].data["feat_pca"][node1_ids]
            feat_v = self.summarized_graph.nodes[ntype].data["feat_pca"][node2_ids]
            
            cu = self.summarized_graph.nodes[ntype].data["node_size"][node1_ids]

            cv = self.summarized_graph.nodes[ntype].data["node_size"][node2_ids]
            self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_u"] = torch.zeros(node1_ids.shape[0], device=self.device)
            self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_v"] = torch.zeros(node2_ids.shape[0], device=self.device)
            du = torch.zeros(node1_ids.shape[0], device=self.device)
            dv = torch.zeros(node2_ids.shape[0], device=self.device)
            
            for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
                if not (ntype == src_type or ntype == dst_type):
                    continue
                du += self.summarized_graph.nodes[ntype].data[f"deg_{etype}"][node1_ids]
                dv += self.summarized_graph.nodes[ntype].data[f"deg_{etype}"][node2_ids]

            feat = (feat_u*cu.unsqueeze(1) + feat_v*cv.unsqueeze(1)) / (cu + cv +du + dv).unsqueeze(1)
            feat_u = (feat_u * cu.unsqueeze(1)) / (du + cu).unsqueeze(1)
            feat_v = (feat_v * cv.unsqueeze(1)) / (dv + cv).unsqueeze(1)
            if self.use_cos_sim:
                self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_u"] += F.cosine_similarity(feat , feat_u, dim=1)
                self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_v"] += F.cosine_similarity(feat ,feat_v, dim=1)
    
            else:
            
        
                self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_u"] += torch.norm(feat - feat_u, p=self.norm_p, dim=1)
                self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_v"] += torch.norm(feat - feat_v, p=self.norm_p, dim=1)
    
    def _sum_costs_feat_in_rgc(self):
        for ntype in self.summarized_graph.ntypes:
            if ntype not in self.merge_graphs.keys():
                continue
            self.merge_graphs[ntype].edata["costs_u"] = torch.zeros(self.merge_graphs[ntype].num_edges(), device=self.device)
            self.merge_graphs[ntype].edata["costs_v"] = torch.zeros(self.merge_graphs[ntype].num_edges(), device=self.device)
        
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            
            self.merge_graphs[src_type].edata["costs_u"] +=  self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"]
            
            
            self.merge_graphs[src_type].edata["costs_v"] +=  self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"] 
            self.metrics['costs_h'][etype].append(torch.mean(self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"] + self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"] )) 

        
            
            
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            
                
            self.merge_graphs[src_type].edata["costs"] = self.merge_graphs[src_type].edata["costs_u"]
            self.merge_graphs[src_type].edata["costs"] += self.merge_graphs[src_type].edata["costs_v"]
            
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            if dst_type not in self.merge_graphs:
                continue
            
            if f"costs_neig_{etype}" in self.merge_graphs[dst_type].edata:
                self.metrics['costs_neigh'][etype].append(torch.mean(self.merge_graphs[dst_type].edata[f"costs_neig_{etype}"])) 
                self.merge_graphs[dst_type].edata["costs"] += self.merge_graphs[dst_type].edata[f"costs_neig_{etype}"]
        for ntype in self.summarized_graph.ntypes:
            self.metrics['costs'][ntype].append(torch.mean(self.merge_graphs[dst_type].edata["costs"])) 
    
    def _sum_costs_feat_sep_with_inner(self):
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            
            self.merge_graphs[src_type].edata["costs_u"] = 2* torch.pow(self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"],2) 
            self.merge_graphs[src_type].edata["costs_u"] -= torch.pow(self.merge_graphs[src_type].edata[f"costs_inner_{etype}_u"], 2)  
            
            
            self.merge_graphs[src_type].edata["costs_v"] = 2* torch.pow(self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"],2) 
            self.merge_graphs[src_type].edata["costs_v"] -= torch.pow(self.merge_graphs[src_type].edata[f"costs_inner_{etype}_v"], 2)  
            
            
        for ntype in self.summarized_graph.ntypes:
            self.merge_graphs[ntype].edata["costs_u"] += 2* torch.pow(self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_u"] , 2)
            self.merge_graphs[ntype].edata["costs_v"] += 2* torch.pow(self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_v"] , 2)
        
            
            
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            self.merge_graphs[src_type].edata["costs"] = torch.sqrt(self.merge_graphs[src_type].edata["costs_u"])
            self.merge_graphs[src_type].edata["costs"] += torch.sqrt(self.merge_graphs[src_type].edata["costs_v"])
            
        
            self.merge_graphs[src_type].edata["costs"] += self.merge_graphs[src_type].edata[f"costs_neig_{etype}"]
    import torch

    def z_score(self, x: torch.Tensor, dim=None, unbiased=False) -> torch.Tensor:
        """
        Compute the z-score of a tensor.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int or tuple of ints, optional): Dimension(s) along which to compute mean and std.
                If None, compute over the entire tensor.
            unbiased (bool): Whether to use Bessel's correction (sample std if True, population std if False).

        Returns:
            torch.Tensor: Tensor of z-scores.
        """
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, unbiased=unbiased, keepdim=True)
        return (x - mean) / std

            
    def _sum_z_score_costs(self):
        
        for ntype in self.summarized_graph.ntypes:
            if ntype not in self.merge_graphs:
                continue
            
            self.merge_graphs[ntype].edata["costs_u"] = torch.zeros(self.merge_graphs[ntype].num_edges(), device=self.device)
            self.merge_graphs[ntype].edata["costs_v"] = torch.zeros(self.merge_graphs[ntype].num_edges(), device=self.device)
            self.merge_graphs[ntype].edata["costs"] = torch.zeros(self.merge_graphs[ntype].num_edges(), device=self.device)

        for ntype in self.summarized_graph.ntypes:
            if ntype not in self.merge_graphs:
                continue
            self.merge_graphs[ntype].edata[f"costs_feat_{ntype}"] = self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_u"] + self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_v"] 
            self.merge_graphs[ntype].edata["costs"] += self.z_score(self.merge_graphs[ntype].edata[f"costs_feat_{ntype}"])
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            if src_type not in self.merge_graphs:
                continue
            self.merge_graphs[src_type].edata["costs_u"] += self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"] 
            
            
            self.merge_graphs[src_type].edata["costs_v"] += self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"]
            
            self.merge_graphs[src_type].edata[f"costs_h_{etype}"] = self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"] + self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"]
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            if dst_type not in self.merge_graphs:
                continue
            
            if f"costs_neig_{etype}" in self.merge_graphs[dst_type].edata:
                if self.multi_relations:
                    etype_around = f"{dst_type}to{src_type}"
                else:
                    etype_around = etype
                self.merge_graphs[dst_type].edata[f"costs_h_{etype_around}"] += self.merge_graphs[dst_type].edata[f"costs_neig_{etype}"]
                
            
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            if src_type not in self.merge_graphs:
                continue
        
            self.merge_graphs[src_type].edata["costs"] += self.z_score(self.merge_graphs[src_type].edata[f"costs_h_{etype}"])
            
        
    
    def _sum_costs_feat_sep(self):
        for ntype in self.summarized_graph.ntypes:
            if ntype not in self.merge_graphs.keys():
                continue
            self.merge_graphs[ntype].edata["costs_u"] = torch.zeros(self.merge_graphs[ntype].num_edges(), device=self.device)
            self.merge_graphs[ntype].edata["costs_v"] = torch.zeros(self.merge_graphs[ntype].num_edges(), device=self.device)
        
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            if src_type not in self.merge_graphs:
                continue
            self.merge_graphs[src_type].edata["costs_u"] += self.merge_graphs[src_type].edata[f"costs_h_{etype}_u"] 
            self.merge_graphs[src_type].edata["costs_v"] += self.merge_graphs[src_type].edata[f"costs_h_{etype}_v"]
                    
            
        for ntype in self.summarized_graph.ntypes:
            if ntype not in self.merge_graphs  or self.cca_cls:
                continue
            #if False :# self.factorin_feat
            self.merge_graphs[ntype].edata["costs_u"] +=  self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_u"] 
            self.merge_graphs[ntype].edata["costs_v"] +=  self.merge_graphs[ntype].edata[f"costs_feat_{ntype}_v"] 
    
        
            
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            if src_type not in self.merge_graphs:
                continue
            
            self.merge_graphs[src_type].edata["costs"] =  self.merge_graphs[src_type].edata["costs_u"]
            self.merge_graphs[src_type].edata["costs"] += self.merge_graphs[src_type].edata["costs_v"]
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            if dst_type not in self.merge_graphs:
                continue
            
            if f"costs_neig_{etype}" in self.merge_graphs[dst_type].edata:
                self.merge_graphs[dst_type].edata["costs"] += self.merge_graphs[dst_type].edata[f"costs_neig_{etype}"]
    
    
    
    def _create_costs(self):
        if self.cca_cls:
            if self.batch_size:
                self._h_cca_costs_batched()
            else:
                self._h_cca_costs()
        else:
            if self.batch_size:
                self._h_costs_batched()
            else:
                self._h_costs()
        self._create_neighbor_costs()
        
        
        if self.feat_in_gcn:
            self._sum_costs_feat_in_rgc()
        else:
            self._feature_costs()    
            if self.use_zscore:
                self._sum_z_score_costs()
            elif self.inner_product:
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
        
        
    
        
      
         

