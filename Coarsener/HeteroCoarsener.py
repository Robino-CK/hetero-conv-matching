import torch
import dgl
from copy import deepcopy
from abc import ABC, abstractmethod
import time
import dgl.function as fn
import numpy as np
from collections import Counter
class HeteroCoarsener(ABC):
    
    
    
    def __init__(self, graph: dgl.DGLHeteroGraph, r:float, num_nearest_init_neighbors_per_type, pairs_per_level=10,approx_neigh= False, add_feat=True, norm_p = 1, device="cpu", use_out_degree=True):
        self.original_graph = graph.to(device)
       # print("lols")
        self.summarized_graph = deepcopy(graph)
        self.summarized_graph = self.summarized_graph.to(device)
        self.approx_neigh = approx_neigh
        self.r = r
        self.norm_p = norm_p
        self.feat_in_gcn = add_feat
        self.device = device
        self.use_out_degree = use_out_degree
        self.num_nearest_init_neighbors_per_type = num_nearest_init_neighbors_per_type
        self.pairs_per_level = pairs_per_level
        self.multi_relations = len(graph.canonical_etypes) > 1
        for ntype in self.summarized_graph.ntypes:
            self.summarized_graph.nodes[ntype].data['node_size'] = torch.ones(self.summarized_graph.num_nodes(ntype), device=self.device)
            
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            self.summarized_graph.edges[etype].data["adj"] = torch.ones(self.summarized_graph.num_edges(etype=etype), device=self.device)
        
        self._update_deg()
        
            
        
        total_num_nodes = graph.num_nodes()
        self.ntype_distribution = dict()
        for ntype in graph.ntypes:
            self.ntype_distribution[ntype] = graph.num_nodes(ntype) / total_num_nodes
        pass
    def opposite_etype(self, etype):
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            if etype == etype:
                return (f"{dst_type}to{src_type}")
    
    
    def _update_deg(self):
            
        rev_sub_g = dgl.reverse(self.summarized_graph,
                        copy_edata=True,      # duplicates every edge feature tensor
                        share_ndata=True)     # node features remain shared views
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            
            if not self.use_out_degree:
                self.summarized_graph.update_all(fn.copy_e('adj', 'm'), fn.sum('m', f'deg_{etype}'), etype=etype)
            
            else:
                rev_sub_g.update_all(fn.copy_e('adj', 'm'), fn.sum('m', f'deg_{etype}'), etype=etype)
            
                self.summarized_graph.nodes[src_type].data[f"deg_{etype}"] = rev_sub_g.nodes[src_type].data[f"deg_{etype}"]
            
            if self.multi_relations:
                
                g = deepcopy(self.summarized_graph)
                g.update_all(fn.copy_e('adj', 'm'), fn.sum('m', f'deg_{etype}'), etype=etype)
                
                rev_sub_g = dgl.reverse(self.summarized_graph,
                        copy_edata=True,      # duplicates every edge feature tensor
                        share_ndata=True)     # node features remain shared views

                rev_sub_g.update_all(fn.copy_e('adj', 'm'), fn.sum('m', f'deg_{etype}'), etype=etype)
             
                self.summarized_graph.nodes[src_type].data[f"deg_{etype}"] = rev_sub_g.nodes[src_type].data[f"deg_{etype}"]
                self.summarized_graph.nodes[dst_type].data[f"deg_{etype}"] = rev_sub_g.nodes[dst_type].data[f"deg_{etype}"]
    
    @abstractmethod
    def _create_gnn_layer(self):
        pass    
    
    def _get_adj(self,nodes_u, nodes_v, etype, g = None):
        if g == None:
            g = self.summarized_graph
        if len(nodes_u) == 0:
            return torch.zeros(0)
        try:
            exists = g.has_edges_between(nodes_u, nodes_v, etype=etype)
        except:
           print(nodes_u, nodes_v) 
           print( self.candidates)
        adj = torch.zeros(len(nodes_u), device=self.device)
        
                       # Get indices where the edge exists
        existing_edge_indices = exists.nonzero(as_tuple=True)[0]


        # Get edge IDs for the existing edges
        edge_ids = g.edge_ids(nodes_u[existing_edge_indices], nodes_v[existing_edge_indices], etype=etype)
        #g_new.nodes[node_type].data[f'i{etype}'][new_nodes] =   infl_uv
    
        edge_data = g.edges[etype].data['adj'][edge_ids]

        # Assign edge data to the result tensor
        adj[existing_edge_indices] = edge_data
        return adj
    
    @abstractmethod
    def _init_merge_graphs(self):
        
        
        pass

    
    def _init_costs(self):
        def _compute_embeddings(node_type, etype=None, use_feat=True):
            # Select the proper feature tensor
            if use_feat and 'feat' in self.summarized_graph.nodes[node_type].data:
                H = self.summarized_graph.nodes[node_type].data['feat'].float()
            elif etype is not None and f'h{etype}' in self.summarized_graph.nodes[node_type].data:
                if self.multi_relations:
                    H = self.summarized_graph.nodes[node_type].data[f'h{etype}'].float()
      
                else:
                    H = self.summarized_graph.nodes[node_type].data[f'SGC{etype}'].float()
      

            H = H.to(self.device)
            return H

        def _query( H):
            # compute pairwise L1 distances
            dist_matrix = torch.cdist(H, H, p=self.norm_p)
            # get k smallest distances (including self)
            k = min(self.num_nearest_init_neighbors_per_type[etype], H.size(0))
            dists, idxs = torch.topk(dist_matrix, k, largest=False)
            return dists, idxs

        def _init_calculate_clostest_edges(src_type, etype, dst_type):
            H = _compute_embeddings(src_type, etype=etype, use_feat=False)
            return _query(H)

        def _init_calculate_clostest_features(ntype):
            H = _compute_embeddings(ntype, use_feat=True)
            return _query(H)
        
        
        start_time = time.time()
        init_costs = dict()
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
           init_costs[etype] = _init_calculate_clostest_edges(src_type, etype, dst_type)
          
        for ntype in self.summarized_graph.ntypes:
            if "feat" in self.summarized_graph.nodes[dst_type].data:
                init_costs[ntype] = _init_calculate_clostest_features(ntype)   
        
       # print("_init_costs", time.time() - start_time)
        return init_costs
    
    def _get_union(self, init_costs):
        start_time = time.time()
        closest = {src: {} for src,_,_ in self.summarized_graph.canonical_etypes}
        # edges
        for src, etype, _ in self.summarized_graph.canonical_etypes:
            dists, idxs = init_costs[etype]
            for i, neighbors in enumerate(idxs):
                neigh = set(neighbors.tolist()) - {i}
                closest[src].setdefault(i, set()).update(neigh)
        # features
        for ntype in self.summarized_graph.ntypes:
            if not ntype in init_costs.keys():
                continue 
            dists, idxs = init_costs[ntype]
            for i, neighbors in enumerate(idxs):
                neigh = set(neighbors.tolist()) - {i}
                closest.setdefault(ntype, {})
                closest[ntype].setdefault(i, set()).update(neigh)
       # print("_get_union", time.time() -start_time)
        type_pairs = dict()
        for typ, nodes in closest.items():
            pairs = [(u, v) for u, vs in nodes.items() for v in vs]
            node1s, node2s = zip(*pairs)
            node1s = torch.tensor(node1s, dtype=torch.long, device=self.device)
            node2s = torch.tensor(node2s, dtype=torch.long, device=self.device)
            
            type_pairs[typ] = torch.stack((node1s, node2s), dim=1 )
        return type_pairs
    
    def _find_lowest_costs(self):
        start_time = time.time()
        topk_non_overlapping_per_type = dict()
        for ntype in self.summarized_graph.ntypes:
            if ntype not in self.merge_graphs:
                continue
            costs = self.merge_graphs[ntype].edata["costs"]
            edges = self.merge_graphs[ntype].edges()
            k = min(self.num_nearest_init_neighbors_per_type[ntype] * self.pairs_per_level * 20, costs.shape[0])
            lowest_costs = torch.topk(costs, k, largest=False, sorted=True)
            
            # Vectorizing the loop
            topk_non_overlapping = []
            nodes = set()
            
            edge_indices = lowest_costs.indices
            src_nodes = edges[0][edge_indices].cpu().numpy()
            dst_nodes = edges[1][edge_indices].cpu().numpy()

            # Avoiding the need to loop over every edge individually
            for i in range(len(src_nodes)):
                if len(nodes) > (self.pairs_per_level * self.ntype_distribution[ntype]):
                    break
                src_node = src_nodes[i]
                dst_node = dst_nodes[i]
                if src_node in nodes or dst_node in nodes:
                    continue
                if src_node == dst_node:
                    continue
                
                topk_non_overlapping.append((src_node, dst_node))
                nodes.add(src_node)
                nodes.add(dst_node)

            topk_non_overlapping_per_type[ntype] = topk_non_overlapping

       # print("_find_lowest_cost_edges", time.time() - start_time)
        return topk_non_overlapping_per_type
    
    def vectorwise_isin(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: (n, d), b: (m, d)
        # Return: (n,) boolean mask
        a_exp = a.unsqueeze(1)        # (n, 1, d)
        b_exp = b.unsqueeze(0)        # (1, m, d)
        matches = (a_exp == b_exp).all(dim=2)  # (n, m)
        
        matches = matches.any(dim=1) 
        a_sym = torch.stack((a[:,1] , a[:,0]), dim=1)
        
        a_exp = a_sym.unsqueeze(1)        # (n, 1, d)
        b_exp = b.unsqueeze(0)        # (1, m, d)
        matches_sym = (a_exp == b_exp).all(dim=2)  # (n, m)
        matches_sym =  matches_sym.any(dim=1) 
        total = torch.logical_or(matches_sym, matches)
        
        return  total 
  
    def _create_mapping(self, a, b):
        assert len(a) == len(b), "mapping needs be one to one "
        mapping = torch.stack((a, b), dim=1) 
                    
        sorted_indices = torch.argsort(mapping[:, 0])
        return mapping[sorted_indices]
                    
                        
               
                
            
    def _create_super_nodes(self, g_before, g_new, mappings):
        self.mappings_temp_v = dict()
        self.mappings_temp_u = dict()
        
        
        for node_type, node_pairs in self.candidates.items():
            mappings[node_type] = torch.arange(0, g_new.num_nodes(ntype=node_type), dtype=torch.int64,device=self.device )
                
            if len(node_pairs) == 0:
                continue
            
            
            
            nodes_u = torch.tensor([i for i, _ in node_pairs], dtype=torch.int64, device=self.device)
            
            nodes_v = torch.tensor([i for  _,i in node_pairs], dtype=torch.int64, device=self.device)
            
            
            num_pairs = len(node_pairs)
            num_nodes_before = g_new.num_nodes(ntype= node_type)
            if "feat" in  g_new.nodes[node_type].data:
                old_feats = g_new.nodes[node_type].data["feat"]
                feat_u = old_feats[nodes_u]
                feat_v = old_feats[nodes_v]
            cu = g_new.nodes[node_type].data["node_size"][nodes_u].unsqueeze(1)
            cv = g_new.nodes[node_type].data["node_size"][nodes_v].unsqueeze(1)
            
            g_new.add_nodes(num_pairs, ntype=node_type)
            
            g_new.nodes[node_type].data["node_size"][num_nodes_before:] = (cu + cv).squeeze()
            if "feat" in  g_new.nodes[node_type].data:
                g_new.nodes[node_type].data["feat"][num_nodes_before:] = (feat_u * cu  + feat_v * cv ) / (cu + cv)
            super_nodes = g_new.nodes(ntype= node_type)[num_nodes_before:]
            
            
            self.mappings_temp_u[node_type] = self._create_mapping(nodes_u, super_nodes)
            self.mappings_temp_v[node_type] = self._create_mapping(nodes_v, super_nodes)
            
            
            
            mappings[node_type][nodes_u] = g_new.nodes(ntype=node_type)[num_nodes_before:]
            mappings[node_type][nodes_v] = g_new.nodes(ntype=node_type)[num_nodes_before:]
            counts_u = (mappings[node_type].unsqueeze(1) > nodes_u).sum(dim=1) 
            counts_v = (mappings[node_type].unsqueeze(1) > nodes_v).sum(dim=1) 
            assert all( mappings[node_type] - counts_u - counts_v > -1), f"mapping wrong '{mappings[node_type]}, {counts_u}, {counts_v}"
            mappings[node_type] = mappings[node_type] - counts_u - counts_v
            assert all(mappings[node_type] > -1) , f"mapping wrong '{mappings[node_type]}, {counts_u}, {counts_v}"
            
            
            feat_u = g_new.nodes[node_type].data["feat"][nodes_u]
            feat_v = g_new.nodes[node_type].data["feat"][nodes_v]
            feat_uv = (feat_u * cu + feat_v * cv) / (cu + cv)
            new_graph_size = g_new.num_nodes(ntype=node_type)
    
            g_new.nodes[node_type].data["feat"][super_nodes] = feat_uv
            
            
            cu = g_new.nodes[node_type].data["node_size"][nodes_u]
            cv = g_new.nodes[node_type].data["node_size"][nodes_v]
            
            
            cuv = cu + cv
            
            g_new.nodes[node_type].data["node_size"][super_nodes] = cuv
        

    def _create_super_nodes_s(self, g_new, node_type, etype, nodes_u, nodes_v, super_nodes):
            
    
            
        du = g_new.nodes[node_type].data[f"deg_{etype}"][nodes_u]
        dv = g_new.nodes[node_type].data[f"deg_{etype}"][nodes_v]
        g_new.nodes[node_type].data[f"deg_{etype}"][super_nodes] = du + dv
               
        su = g_new.nodes[node_type].data[f's{etype}'][nodes_u]  
        sv =  g_new.nodes[node_type].data[f's{etype}'][nodes_v] 
        if self.multi_relations:
            g_new.nodes[node_type].data[f"s{etype}"][super_nodes] = su + sv
            return
    
        
        adj_uv = self._get_adj(nodes_u, nodes_v, etype)
        adj_vu = self._get_adj(nodes_v, nodes_u, etype)
        
        feat_u = g_new.nodes[node_type].data["feat"][nodes_u]
        feat_v = g_new.nodes[node_type].data["feat"][nodes_v]
        cu = g_new.nodes[node_type].data["node_size"][nodes_u]
        cv = g_new.nodes[node_type].data["node_size"][nodes_v]
        suv = su - (adj_vu / (torch.sqrt(du + cu ))).unsqueeze(1) * feat_u  + sv -   (adj_uv / (torch.sqrt(dv + cv ))).unsqueeze(1)  * feat_v
        g_new.nodes[node_type].data[f"s{etype}"][super_nodes] = suv
    
    def _get_supernode_indices(self, base_nodes, query_nodes, super_nodes):
        
        sorted_mapping = self._create_mapping(base_nodes, super_nodes)
        indices = torch.searchsorted(sorted_mapping[:, 0], query_nodes)
        return sorted_mapping[indices, 1]
        
        
    
        
        
    def _connect_super_nodes_to_out_edges(self, g_before, g_new, nodes, super_nodes, etype):
        # connect super nodes out edges to nodes
        src, dst, eid = g_before.out_edges(nodes, form="all", etype=etype)
        #adj = g_new.edges[etype].data["adj"][eid]
        adj = self._get_adj(src, dst, etype)
        super_nodes_src = self._get_supernode_indices(nodes, src, super_nodes)
        
        g_new.add_edges( super_nodes_src, dst,data={"adj": adj}, etype=etype)
    
    def _connect_super_nodes_to_in_edges(self, g_before, g_new, nodes, super_nodes, etype): 
         
        # connect super nodes in edges to nodes
        src, dst, eid = g_before.in_edges(nodes, form="all", etype=etype)
        #adj = g_new.edges[etype].data["adj"][eid]
        adj = self._get_adj(src, dst, etype)
        
        super_nodes_dst = self._get_supernode_indices(nodes, dst, super_nodes)
        
        g_new.add_edges( src, super_nodes_dst,data={"adj": adj}, etype=etype)

    def _connect_super_nodes_to_supernodes_single_etype(self, g_before, g_new, nodes,   super_nodes, etype):
        src, dst, eid = g_new.in_edges(super_nodes, form="all", etype=etype)
        #adj = g_new.edges[etype].data["adj"][eid]
        
        
        mask_src = torch.isin(src, nodes)
      #  mask_dst = torch.isin(dst, super_nodes_src)
      #  mask = torch.logical_and(mask_src, mask_dst)
        src = src[mask_src]
        super_nodes_src_2 = self._get_supernode_indices(nodes, src, super_nodes)
        
        
        dst = dst[mask_src]
       
       # super_nodes_dst = self._get_supernode_indices(nodes_src, dst, super_nodes_src)
        
        adj = self._get_adj(src,dst , etype, g_new)
        
        g_new.add_edges( super_nodes_src_2, dst,data={"adj": adj}, etype=etype)
    
    def _connect_super_nodes_to_supernodes(self, g_before, g_new, nodes_src, nodes_dst,  super_nodes_src, super_nodes_dst, etype):
        src, dst, eid = g_before.in_edges(nodes_dst, form="all", etype=etype)
        adj = g_new.edges[etype].data["adj"][eid]
        mask = torch.isin(src, nodes_src)
        src = src[mask]
        super_nodes_src = self._get_supernode_indices(nodes_src, src, super_nodes_src)
        
        adj = g_new.edges[etype].data["adj"][eid]
        dst = dst[mask]
        super_nodes_dst = self._get_supernode_indices(nodes_dst, dst, super_nodes_dst)
        
        
        g_new.add_edges( super_nodes_src, super_nodes_dst,data={"adj": adj}, etype=etype)
    
        
    
    def _create_full_g(self, g_before,g_new, mappings):
        self._create_super_nodes(g_before, g_new, mappings)
            
        for node_type, node_pairs in self.candidates.items():
            if len(node_pairs) == 0:
                    continue
            for src_type, etype,dst_type in g_new.canonical_etypes:
                if node_type == src_type:
                    node_type = src_type
                else:
                    continue
                nodes_u = torch.tensor([i for i, _ in node_pairs], dtype=torch.int64, device=self.device)
                
                nodes_v = torch.tensor([i for  _,i in node_pairs], dtype=torch.int64, device=self.device)
                
                super_nodes = g_new.nodes(ntype=node_type)[g_before.num_nodes(ntype=node_type):]
                self._create_super_nodes_s(g_new,node_type, etype, nodes_u, nodes_v, super_nodes)
                
            
        for node_type, node_pairs in self.candidates.items():
            if len(node_pairs) == 0:
                continue        
            nodes_u = torch.tensor([i for i, _ in node_pairs], dtype=torch.int64, device=self.device)
                
            nodes_v = torch.tensor([i for  _,i in node_pairs], dtype=torch.int64, device=self.device)
            super_nodes = g_new.nodes(ntype=node_type)[g_before.num_nodes(ntype=node_type):]
                
            for src_type, etype,dst_type in g_new.canonical_etypes:
                if node_type == src_type: #  
                    self._connect_super_nodes_to_out_edges(g_before, g_new, nodes_u, super_nodes, etype)
                    self._connect_super_nodes_to_out_edges(g_before, g_new, nodes_v, super_nodes, etype)
                    
                if  node_type == dst_type:
                    self._connect_super_nodes_to_in_edges(g_before, g_new, nodes_u, super_nodes, etype)
                    self._connect_super_nodes_to_in_edges(g_before, g_new, nodes_v, super_nodes, etype)
        # if self.add_feat:
        #     return
        for src_type, etype,dst_type in g_new.canonical_etypes:
            if self.candidates[src_type] == [] or self.candidates[dst_type] == []:  
                continue
            
            node_pairs = self.candidates[src_type]
            
            nodes_u_src = torch.tensor([i for i, _ in node_pairs], dtype=torch.int64, device=self.device)
                
            nodes_v_src = torch.tensor([i for  _,i in node_pairs], dtype=torch.int64, device=self.device)
            super_nodes_src = g_new.nodes(ntype=src_type)[g_before.num_nodes(ntype=src_type):]
            super_nodes_dst = g_new.nodes(ntype=dst_type)[g_before.num_nodes(ntype=dst_type):]
            
            node_pairs = self.candidates[dst_type]
            
            nodes_u_dst = torch.tensor([i for i, _ in node_pairs], dtype=torch.int64, device=self.device)
                
            nodes_v_dst = torch.tensor([i for  _,i in node_pairs], dtype=torch.int64, device=self.device)
            if src_type == dst_type:
                self._connect_super_nodes_to_supernodes_single_etype(g_before, g_new, nodes_u_src, super_nodes_src,  etype)
                self._connect_super_nodes_to_supernodes_single_etype(g_before, g_new, nodes_v_src, super_nodes_src,  etype)
                
            else:
                
                self._connect_super_nodes_to_supernodes(g_before, g_new, nodes_u_src, nodes_u_dst, super_nodes_src, super_nodes_dst, etype)
                self._connect_super_nodes_to_supernodes(g_before, g_new, nodes_v_src, nodes_v_dst, super_nodes_src, super_nodes_dst, etype)
            

    
    def _merge_nodes(self, g_before):
            """
      
            """
            start_time = time.time()
            g_new = deepcopy(g_before)
            mappings = dict()
            self._create_full_g(g_before, g_new, mappings)
            for src_type, etype,dst_type in g_new.canonical_etypes:
                    #if node_type != dst_type:
                     #       continue
                    node_pairs = self.candidates[dst_type]
                    node_type = dst_type
                #for node_type, node_pairs in self.candidates.items():
                    if len(node_pairs)== 0:
                        continue
                    nodes_u = torch.tensor([i for i, _ in node_pairs], dtype=torch.int64, device=self.device)
                    nodes_v = torch.tensor([i for  _,i in node_pairs], dtype=torch.int64, device=self.device)
 
                    num_nodes_before = g_before.num_nodes(ntype= node_type)
                    super_nodes = g_new.nodes(ntype= node_type)[num_nodes_before:]
                
                    g_new.nodes[node_type].data[f"deg_{etype}"][super_nodes] = g_new.nodes[node_type].data[f"deg_{etype}"][nodes_u] + g_new.nodes[node_type].data[f"deg_{etype}"][nodes_v]
               
                    nodes_uv = torch.cat([nodes_u, nodes_v])
                    # if self.add_feat:
                    #     g_new.nodes[node_type].data[f"i{etype}"][super_nodes] = g_new.nodes[node_type].data[f"i{etype}"][nodes_u] + g_new.nodes[node_type].data[f"i{etype}"][nodes_v] 
                    def msg_minus_neigh_s(edges):
                        return {'s':  (edges.data['adj'].unsqueeze(1) *edges.src["feat"])/torch.sqrt(edges.src[f"deg_{etype}"] + edges.src['node_size']).unsqueeze(1) ,
                                'i':edges.data['adj']/torch.sqrt(edges.src[f"deg_{etype}"] + edges.src['node_size']) ,
                                
                                
                                } #

                    def reduce_minus_neigh_s(nodes):
                        return {f's_new': torch.sum(nodes.mailbox['s']  , dim=1), f'i_new': torch.sum(nodes.mailbox['i']  , dim=1)}
                    # update Neigbors
                    edges_src, edges_dst = g_new.in_edges(nodes_uv,  etype=etype)
                    if True:#self.multi_relations: #and self.feat_in_gcn:
                        edges = torch.stack((edges_src, edges_dst), dim=1)
                        nodes_u_supernodes = torch.stack((nodes_u, super_nodes), dim = 1)
                        nodes_v_supernodes = torch.stack((nodes_v, super_nodes), dim = 1)
                        mask_u = self.vectorwise_isin(edges, nodes_u_supernodes) == False
                        mask_v = self.vectorwise_isin(edges, nodes_v_supernodes) == False
                        mask = torch.logical_and(mask_u, mask_v)
                        edges_src = edges_src[mask]
                        edges_dst = edges_dst[mask]
                        
                    if len(edges_dst) > 0:
                    
                        rev_sub_g = dgl.reverse(g_new,
                        copy_edata=True,      # duplicates every edge feature tensor
                        share_ndata=True)     # node features remain shared views
                    
                        rev_sub_g.send_and_recv((edges_dst,edges_src ), message_func=msg_minus_neigh_s, reduce_func=reduce_minus_neigh_s, etype=etype)
                        g_new.nodes[src_type].data[f"s{etype}"] -= rev_sub_g.nodes[src_type].data["s_new"]
                        # if self.add_feat:
                        #     g_new.nodes[node_type].data[f"i{etype}"] -=  rev_sub_g.nodes[src_type].data["i_new"]
                    
                    # update U,V
                    edges_src, edges_dst = g_new.in_edges(super_nodes,  etype=etype)
                    if len(edges_dst) > 0:
                    
                    
                        rev_sub_g = dgl.reverse(g_new,
                            copy_edata=True,      # duplicates every edge feature tensor
                            share_ndata=True)     # node features remain shared views
                        rev_sub_g.send_and_recv((edges_dst,edges_src ), message_func=msg_minus_neigh_s, reduce_func=reduce_minus_neigh_s, etype=etype)
                        g_new.nodes[src_type].data[f"s{etype}"] += rev_sub_g.nodes[src_type].data["s_new"]
                        # if self.add_feat:
                        #     g_new.nodes[node_type].data[f"i{etype}"] += rev_sub_g.nodes[src_type].data["i_new"]
                    

                    
                    
                    
                    
            for node_type, node_pairs in self.candidates.items():
                if len(node_pairs) == 0:
                    continue
                nodes_u = torch.tensor([i for i, _ in node_pairs], dtype=torch.int64, device=self.device)
                nodes_v = torch.tensor([i for  _,i in node_pairs], dtype=torch.int64, device=self.device)
                
                nodes_to_delete = torch.cat((nodes_u, nodes_v))           
                g_new.remove_nodes(nodes_to_delete, ntype=node_type)
            
            for node_type, node_pairs in self.candidates.items():
                
                for src_type, etype,dst_type in g_new.canonical_etypes:
                    if node_type != src_type:
                        continue
                    
                    mapping_src = mappings[src_type]
                    mapping_dst = mappings[dst_type]
                    all_eids = g_new.edges(form='eid', etype=etype)
                    
                    
                    
                    g_new.remove_edges(all_eids, etype=etype)
                    edges_original = g_before.edges(etype=etype)
                    edges_adj = g_before.edges[etype].data["adj"]
                    
                    new_edges = torch.stack((mapping_src[edges_original[0]], mapping_dst[edges_original[1]]))
                    pairs = torch.stack((new_edges[0], new_edges[1]), dim=1)
                    
                    
                    uniq_pairs, inverse = torch.unique(
                    pairs, dim=0, return_inverse=True
                        )
                    sums = torch.zeros(len(uniq_pairs), dtype=edges_adj.dtype, device=self.device)
                    sums.index_add_(0, inverse, edges_adj) 
                    
                    eids = g_new.add_edges(uniq_pairs[:,0], uniq_pairs[:,1], etype=(src_type, etype, dst_type))
                    g_new.edges[etype].data["adj"][eids] = sums
                    self._update_deg()   
            
            
                    # if src_type == dst_type:
                    #     g_new = dgl.remove_self_loop(g_new, etype=etype)
                    
                    c = g_new.nodes[src_type].data[f"node_size"] 
                    d = g_new.nodes[src_type].data[f"deg_{etype}"] 
                    f = g_new.nodes[src_type].data[f"feat"]
                    s = g_new.nodes[src_type].data[f"s{etype}"]
                    if self.feat_in_gcn:
                        g_new.nodes[src_type].data[f"h{etype}"] = ((c / (c + d ) ).unsqueeze(1) * f) + (1 / torch.sqrt(c + d)).unsqueeze(1) * s
                    else:
                        g_new.nodes[src_type].data[f"h{etype}"] = (1 / torch.sqrt(c + d)).unsqueeze(1) * s
                    
                        
                    
                                 
           # print("_merge_nodes", time.time() - start_time)
            return g_new, mappings


            
    
   # @abstractmethod
    def reduce_merge_graph(self, mappings):
        for node_type, node_pairs in self.candidates.items():
            
            mapping = mappings[node_type]
           
            merge_graph_copy = deepcopy(self.merge_graphs[node_type])
            all_eids = merge_graph_copy.edges(form='eid')
            
            self.merge_graphs[node_type].remove_edges(all_eids)
            edges_original = merge_graph_copy.edges()
            
            
            new_edges = torch.stack((mapping[edges_original[0]], mapping[edges_original[1]]))
            pairs = torch.stack((new_edges[0], new_edges[1]), dim=1)
            
            
            uniq_pairs = torch.unique( pairs, dim=0)
           
            
            self.merge_graphs[node_type].add_edges(uniq_pairs[:,0], uniq_pairs[:,1])
            self.merge_graphs[node_type] = self.merge_graphs[node_type].remove_self_loop()


    
                    
    def get_labels(self, mapping, ntype):
        
        labels_dict = dict()
        inverse_mapping = dict()
        for ori_node, coar_node in mapping.items():
            if coar_node in inverse_mapping:
                inverse_mapping[coar_node].append(ori_node)
            else:
                inverse_mapping[coar_node] = [ori_node]
        for coar_node, ori_list in inverse_mapping.items():
            label_list = []
            for ori_node in ori_list:
                label_list.append(self.original_graph.nodes[ntype].data["label"][ori_node].item())
            counter = Counter(label_list)
            
            labels_dict[coar_node],_ = counter.most_common()[0]
        
        return labels_dict

    
    def get_mapping(self, ntype):
        master_mapping = dict()
        nodes_orig = self.original_graph.nodes(ntype)
        nodes = self.original_graph.nodes(ntype)
        for mapping in self.mappings:
            nodes = mapping[ntype][nodes]
        for i in range(len(nodes)):
            master_mapping[nodes_orig[i].item()] = nodes[i].item()
        
        return master_mapping 
    
    
    def init(self):
        self.mappings = [] 
        
        if not self.multi_relations:
            self._create_sgn_layer()
        self._create_gnn_layer()
        init_costs = self._init_costs()
        type_pairs = self._get_union(init_costs)
        self._init_merge_graphs(type_pairs)
        self.candidates = self._find_lowest_costs()
       
    @abstractmethod
    def _update_merge_graph(self, mappings):
        pass
    def coarsen_step(self):
        self.summarized_graph, mappings = self._merge_nodes(self.summarized_graph)
        self.mappings.append(mappings)
        self._update_merge_graph(mappings)
        
    