
import torch
import dgl
from Coarsener.HeteroCoarsener import HeteroCoarsener
import time
import dgl.backend as F         # this is the backend (torch, TF, etc.)


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
            c = self.summarized_graph.nodes[src_type].data['node_size']
            
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
        h_all = ((su + sv) - minus )/ torch.sqrt((deg1 + deg2 + cuv.squeeze())).unsqueeze(1 ) + feat * ( cuv.squeeze() / ((deg1 + deg2 + cuv.squeeze()))).unsqueeze(1)  #+   #)  # (L, D)

        return h_all
    
    def _create_neighbor_costs(self):
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            
            merge_graph = self.merge_graphs[src_type] 
            merge_node_u, merge_node_v, merge_graph_eid = merge_graph.edges(form="all")
            
            
            
            
            sum_graph_src_from_merge_graph, merge_pairs_dst, neigbors_ei = self.summarized_graph.in_edges(merge_node_u,form = "all",etype=etype)
            
            
            h = self.summarized_graph.nodes[src_type].data[f"h{etype}"][merge_pairs_dst]
            ci = self.summarized_graph.nodes[src_type].data["node_size"][merge_pairs_dst]
            di = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"][merge_pairs_dst]
            fi = self.summarized_graph.nodes[src_type].data[f"feat"][merge_pairs_dst]
            fi = self.summarized_graph.nodes[src_type].data[f"s{etype}"][merge_pairs_dst]
            h_prim = (ci / (ci + di)).unsqueeze(1) * fi + (1 / torch.sqrt(di + ci)).unsqueeze(1) * fi
            
            
            
            
            mapping_merge_src_dst = self._create_mapping(merge_node_u, merge_node_v)
            cu = self.summarized_graph.nodes[src_type].data["node_size"][sum_graph_src_from_merge_graph]
            du = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"][sum_graph_src_from_merge_graph]
            fu = self.summarized_graph.nodes[src_type].data[f"feat"][sum_graph_src_from_merge_graph]
            a_ui = self._get_adj(sum_graph_src_from_merge_graph, merge_pairs_dst, etype=etype)
            #a_u_i = 
            
            indices = torch.searchsorted(mapping_merge_src_dst[:,0], merge_pairs_dst)
            result = mapping_merge_src_dst[indices, 1]
            self.summarized_graph.nodes[src_type].data["node_size"][result]
           #
            cv = self.summarized_graph.nodes[src_type].data["node_size"][result]
            dv = self.summarized_graph.nodes[src_type].data[f"deg_{etype}"][result]
            fv = self.summarized_graph.nodes[src_type].data[f"feat"][result]
            
            f_uv = (cu.unsqueeze(1) * fu + fv * cv.unsqueeze(1) ) / (cu + cv).unsqueeze(1)
            a_vi = self._get_adj(merge_pairs_dst, result, etype=etype)
            
             
            h_prim += (a_ui + a_vi).unsqueeze(1) / torch.sqrt( (di + ci) * (du + dv + cu + cv)).unsqueeze(1)  * f_uv
            h_prim -= (a_ui).unsqueeze(1) / torch.sqrt( (di + ci) * (du +cu )).unsqueeze(1)  * fu
            h_prim -= (a_vi).unsqueeze(1) / torch.sqrt( (di + ci) * (dv +cv )).unsqueeze(1)  * fv
            
          
            costs = torch.norm(h - h_prim, p = 1, dim = 1)
            mapping_merge_node_merge_eid = self._create_mapping(merge_node_u, merge_graph_eid)
            
            mapping_sumg_mg_src_to_sum_eid = self._create_mapping(sum_graph_src_from_merge_graph, neigbors_ei)
            indices = torch.searchsorted(mapping_merge_node_merge_eid[:,0], sum_graph_src_from_merge_graph) 
            result = mapping_merge_node_merge_eid[indices, 1]
               
            merge_graph.edata["costs"][merge_graph_eid][result] += costs
            
        pass
    
        
    def _h_costs(self,type_pairs=None):
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
            

            node1_ids = type_pairs[src_type][:,0]  # [P]
            node2_ids = type_pairs[src_type][:,1]    # [P]

            # gather representations
            h1 = self.summarized_graph.nodes[src_type].data[f"h{etype}"][node1_ids]  # [P, H]
            h2 = self.summarized_graph.nodes[src_type].data[f"h{etype}"][node2_ids]  # [P, H]
            # build a dense [num_src, hidden] tensor
            #H_tensor =  torch.tensor([v for k,v in  H_merged.items()] , device=device)
            merged = H_merged                               # [P, H]
            
            
            
            
            # L1 costs
            cost = torch.norm(merged - h1, p=1, dim=1) + torch.norm(merged - h2, p=1, dim=1)
            
            self.merge_graphs[src_type].edata["costs"] = cost 
            #
                
    def _init_merge_graphs(self, type_pairs):
        self.merge_graphs = dict()
        self._h_costs( type_pairs)
        self._create_neighbor_costs()
            
        
        
    def _update_merge_graph(self, type_pairs):
        self._update_h_costs( type_pairs)
        
    
        
      
         

