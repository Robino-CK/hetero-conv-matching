
import torch
import dgl
from Coarsener.HeteroCoarsener import HeteroCoarsener
import time
import dgl.backend as F         # this is the backend (torch, TF, etc.)
from torch_scatter import scatter_add   
import numpy as np
class HeteroRGCNCoarsener(HeteroCoarsener):
    
    def _create_sgn_layer(self, k =1):
        for src_type, etype, _ in self.summarized_graph.canonical_etypes:
            g = dgl.add_self_loop(self.summarized_graph, etype=etype)
            A = g.adj_external(etype=etype).to_dense()
            D = torch.diag(torch.rsqrt(torch.sum(A, dim=1)))
            feat =  self.summarized_graph.nodes[src_type].data['feat']
            H =  torch.pow((D @A @  D), k ) @ feat
            self.summarized_graph.nodes[src_type].data[f"SGC{etype}"] = H
       
        
    def _create_gnn_layer(self, k = 1):
        
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
                
            
            s_e = feat_v * inv_sqrt_out[v].unsqueeze(-1)       # [E, D]
            # Scatter-add to compute S at source nodes
           
            S_tensor = torch.zeros((n_src, feat_dim), device=self.device)
            S_tensor = S_tensor.index_add(0, u, s_e)
            infl = torch.zeros(n_src, device=self.device)
            infl = infl.index_add(0, v, inv_sqrt_in[u])

            # Compute H = D_out^{-1/2} * S
            H_tensor = inv_sqrt_out.unsqueeze(-1) * S_tensor

            # Store in summarized_graph
            self.summarized_graph.nodes[src_type].data[f'i{etype}'] = infl
            
            self.summarized_graph.nodes[src_type].data[f's{etype}'] = S_tensor
           
            self.summarized_graph.nodes[src_type].data[f'h{etype}'] = H_tensor + ((feats / (deg_out + c).unsqueeze(1) ))
           

        print("_create_h_spatial_rgcn", time.time() - start_time)

    
    def _create_h_merged(self,  node1s, node2s,ntype, etype):
        cache = self.summarized_graph.nodes[ntype].data[f's{etype}']
        
        # 1) Flatten your table into two 1-D lists of equal length L:
        node1s = torch.tensor(node1s, dtype=torch.long, device=self.device)
        node2s = torch.tensor(node2s, dtype=torch.long, device=self.device)

        # 2) Grab degrees and caches in one go:
        
        deg =  self.summarized_graph.nodes[ntype].data[f"deg_{etype}"]
       
        deg1 = deg[node1s]  # shape (L,)
        deg2 = deg[node2s]            # shape (L,)

        
        su = cache[node1s]            # shape (L, D)
        sv = cache[node2s]
        
                
        adj1 = self._get_adj(node1s, node2s, etype)
        adj2 = self._get_adj(node2s, node1s, etype)
        
        feat_u = self.summarized_graph.nodes[ntype].data["feat"][node1s]
        feat_v = self.summarized_graph.nodes[ntype].data["feat"][node2s]
        
        
        cu = self.summarized_graph.nodes[ntype].data["node_size"][node1s].unsqueeze(1)
        cv = self.summarized_graph.nodes[ntype].data["node_size"][node2s].unsqueeze(1)
        minus = torch.mul((adj2 / torch.sqrt(deg1 + cu.squeeze() )).unsqueeze(1), feat_u) + torch.mul((adj1 / torch.sqrt(deg2 + cv.squeeze() )).unsqueeze(1), feat_v) # + torch.matmul( (adj / torch.sqrt(deg2 + cv.squeeze() )), feat_v)
        
        

        # 3) Cluster‐size term (make sure cluster_sizes is a tensor):
      
        cuv = cu + cv  # shape (L,)

        # 4) Single vectorized compute of h for all L pairs:
        #    (we broadcast / unsqueeze cuv into the right D-dimensional form)
        feat = (feat_u*cu + feat_v*cv) / (cu + cv)
        plus =     torch.mul((adj2 / torch.sqrt(deg1 + deg2 + cuv.squeeze() )).unsqueeze(1), feat) + torch.mul((adj1 / torch.sqrt(deg1 + deg2 + cuv.squeeze() )).unsqueeze(1), feat)
        h_all =  feat * ( cuv.squeeze() / ((deg1 + deg2 + cuv.squeeze()))).unsqueeze(1) + (su + sv - minus + plus)/ torch.sqrt((deg1 + deg2 + cuv.squeeze())).unsqueeze(1 )  #+   #)  # (L, D)

        return h_all
    
    def neigbor_approx_difference_per_pair(self,g, pairs,  d_node, c_node, infl_node, feat_node, etype):
        src, dst = g.edges(etype=etype)
        edges = torch.stack((src, dst), dim=1)  # (E, 2)
        u, v = pairs[:,0], pairs[:,1]
        mask = True#self.vectorwise_isin(pairs, edges) == False
        
        adj = self._get_adj(u,v , etype=etype) + self._get_adj(v,u , etype=etype)
   #     adj_vu = self._get_adj(v,u , etype=etype)
        
        cu = c_node[u]
        cv = c_node[v]
        
        du = d_node[u]
        dv = d_node[v]
        
        iu = infl_node[u] 
        iv = infl_node[v]
        
        feat_u = feat_node[u]
        feat_v = feat_node[v]
        
        feat = (feat_u*cu.unsqueeze(1) + feat_v*cv.unsqueeze(1)) / (cu + cv).unsqueeze(1)
        
        
        neigh_cost_u = torch.norm( (feat / torch.sqrt(du + dv + cu + cv).unsqueeze(1)   - (feat_u / torch.sqrt(du + cu).unsqueeze(1))) , dim=1, p=1)   * iu
        neigh_cost_v = torch.norm( (feat / torch.sqrt(du + dv + cu + cv).unsqueeze(1)   - (feat_v / torch.sqrt(dv + cv).unsqueeze(1))), dim=1, p=1)   * iv
        #res = torch.where(mask,  neigh_cost_u + neigh_cost_v, torch.tensor(0.0, device=self.device))
        return neigh_cost_u + neigh_cost_v
    
    def neighbor_difference_per_pair(self,g, pairs, h_node, d_node, c_node, s_node, feat_node, a_edge, etype):
        """
        For every pair (u,v) and every neighbour n of u or v compute

            d_{(u,v),n} = [h_n + a_{un} h_u + a_{vn} h_v] - m_n

        Returns
        -------
        d            : (K, F_h)  one row per (pair, neighbour)
        pair_d_sum   : (P, F_h)  sum over neighbours for each pair
        """
        
        start_time = time.time()   
        
        device = h_node.device
        pairs  = pairs.to(device)
        u, v   = pairs[:, 0], pairs[:, 1]          # (P,)
        P      = pairs.shape[0]

        # ------------------------------------------------------------------ #
        # 1 ▸ pick all edges touching any u or v                             #
        # ------------------------------------------------------------------ #
        src, dst, _ = g.edges(form='all', etype=etype)
        src, dst    = src.to(device), dst.to(device)
        print("get edges",  time.time()  - start_time )
        start_time = time.time()   
        
        touches = (
         (dst[None, :] == u[:, None]) |      # dst is u
                  (dst[None, :] == v[:, None])        # dst is v
            )
        print("create touch",  time.time()  - start_time )
        
        start_time = time.time()   
        
        pair_idx, edge_idx = touches.nonzero(as_tuple=True)      # (K,)

        e_src, e_dst      = src[edge_idx], dst[edge_idx]
        a_val             = a_edge[edge_idx].view(-1)            # (K,)

        nbr               =  src[edge_idx]       #torch.where(is_src_endpoint, e_dst, e_src)

        connects_u        = (dst[edge_idx] == u[pair_idx])
        connects_v        =  (dst[edge_idx] == v[pair_idx])                                
        print("create connection",  time.time()  - start_time )
        # ------------------------------------------------------------------ #
        # 2 ▸ heterograph: pair ─knows─> neighbour                          #
        # ------------------------------------------------------------------ #
     #   t = torch.stack((pair_idx)
        hg = dgl.heterograph(
            {('pair', 'knows', 'node'): (pair_idx, nbr)},
            
            num_nodes_dict={'pair': P, 'node': g.num_nodes()}
        )

        hg.nodes['pair'].data['id_u'] = u
        hg.nodes["pair"].data["id_v"] = v
        hg.nodes['pair'].data['f_u'] = feat_node[u]
        hg.nodes['pair'].data['f_v'] = feat_node[v]
        
        hg.nodes['pair'].data['d_u'] = d_node[u]
        hg.nodes['pair'].data['d_v'] = d_node[v]
        
        hg.nodes['pair'].data['c_u'] = c_node[u]
        hg.nodes['pair'].data['c_v'] = c_node[v]
        
        
        
        hg.nodes["node"].data["id_n"] = torch.arange(d_node.shape[0], device=self.device)
        hg.nodes['node'].data['d_n'] = d_node
        hg.nodes['node'].data['c_n'] = c_node
        hg.nodes['node'].data['s_n'] = s_node
        hg.nodes['node'].data['f_n'] = feat_node
        hg.nodes['node'].data['h_n'] = h_node


        hg.edges['knows'].data['a_un'] = (a_val) * connects_u.float()
        hg.edges['knows'].data['a_vn'] = (a_val )* connects_v.float()
        hg.edges['knows'].data['a_uvn'] = (a_val) * (connects_u.float() +connects_v.float())
    #    hg.edges['knows'].data[''] = (a_val )* connects_v.float()
        start_time = time.time()
        hg = hg.to("cpu").to_simple(hg,  aggregator="sum", copy_edata=True)
        hg = hg.to(self.device)
        print("to simple", time.time() -start_time)
        
        # ------------------------------------------------------------------ #
        # 3 ▸ fused CUDA kernel per (pair,nbr) edge                          #
        # ------------------------------------------------------------------ #
        #mask = torch.logical_and(hg.nodes['pair'].data["id_u"] == 2, hg.nodes['pair'].data["id_v"] == 4)
        def edge_formula(edges):
            ##mask = torch.logical_and(edges.src["id_u"] == 2, edges.src["id_v"] == 4)
            mask =  torch.logical_or(edges.dst["id_n"] == edges.src['id_u'], edges.dst["id_n"] == edges.src['id_v'])
            mask = mask == False
            h_prime = (edges.dst["c_n"] / (edges.dst["d_n"] + edges.dst["c_n"])).unsqueeze(1) * edges.dst["f_n"]
            h_prime += (1 / torch.sqrt(edges.dst["d_n"] + edges.dst["c_n"])).unsqueeze(1) * edges.dst["s_n"]
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
            
            d = torch.norm(h_prime - edges.dst['h_n'], p=1, dim=1)
            d = torch.where(mask, d, torch.tensor(0))
            return {'d': d}
        start_time = time.time()
        hg.apply_edges(edge_formula, etype=('pair', 'knows', 'node'))

        d = hg.edges['knows'].data['d']                   # (K, F_h)

        # ------------------------------------------------------------------ #
        # 4 ▸ aggregate d over neighbours for each pair                      #
        # ------------------------------------------------------------------ #
        pair_idx,_ = hg.edges(etype="knows")
        pair_d_sum = scatter_add(d, pair_idx, dim=0, dim_size=P) 
        print("edges & scatter", time.time() -start_time)
        # (P, F_h)

        return pair_d_sum

    
    
    def _create_neighbor_costs(self):
        start_time = time.time()
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            
            merge_graph = self.merge_graphs[src_type] 
            merge_node_u, merge_node_v, merge_graph_eid = merge_graph.edges(form="all")
            pairs = torch.stack((merge_node_u, merge_node_v) , dim=1)
            h_node = self.summarized_graph.nodes[src_type].data[f"h{etype}"]
            d_node = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"]
            c_node = cv = self.summarized_graph.nodes[src_type].data["node_size"]
            s_node =  self.summarized_graph.nodes[src_type].data[f"s{etype}"]
            feat_node = self.summarized_graph.nodes[src_type].data[f"feat"]
            infl_node = self.summarized_graph.nodes[src_type].data[f"i{etype}"]
          
            a_edge = self.summarized_graph.edges[etype].data[f"adj"]
            if self.approx_neigh:
                neighbors_cost  = self.neigbor_approx_difference_per_pair(self.summarized_graph, pairs,d_node, c_node, infl_node, feat_node, etype)
            else:            
                neighbors_cost  = self.neighbor_difference_per_pair(self.summarized_graph, pairs, h_node, d_node, c_node, s_node, feat_node, a_edge, etype)
            merge_graph.edata["costs_neig"] = neighbors_cost
        print("_create_neighbor_costs", time.time() - start_time)
    
    
    
    
        
    def _h_costs(self,type_pairs=None):
        start_time = time.time()
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            # ensure nested dict
            if type_pairs:
                self.merge_graphs[src_type] = dgl.graph(([], []), num_nodes=self.summarized_graph.number_of_nodes(ntype=src_type), device=self.device)
              # self.merge_graphs[src_type] = dgl.to_bidirected(self.merge_graphs[src_type])
                self.merge_graphs[src_type].add_edges(type_pairs[src_type][:,0],type_pairs[src_type][:,1])
                
                
            
            
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
            
            
            
            
            # L1 costs
            cost = torch.norm(merged - h1, p=1, dim=1) + torch.norm(merged - h2, p=1, dim=1)
            
            self.merge_graphs[src_type].edata["costs_h"] = cost 
        print("_h_costs", time.time() - start_time)
                
    def _init_merge_graphs(self, type_pairs):
        self.merge_graphs = dict()
        self._h_costs( type_pairs)
        self._create_neighbor_costs()
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            self.merge_graphs[src_type].edata["costs"] = self.merge_graphs[src_type].edata["costs_h"] + self.merge_graphs[src_type].edata["costs_neig"] 
            
        
        
    def _update_merge_graph(self, mappings):
        self.reduce_merge_graph(mappings)
        self._h_costs()
        self._create_neighbor_costs()
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            self.merge_graphs[src_type].edata["costs"] = self.merge_graphs[src_type].edata["costs_h"] + self.merge_graphs[src_type].edata["costs_neig"] 
        self.candidates = self._find_lowest_costs()
        
        
    
        
      
         

