import torch
import dgl
from copy import deepcopy
from abc import ABC, abstractmethod
import time
import dgl.function as fn
import numpy as np
class HeteroCoarsener(ABC):
    def __init__(self, graph: dgl.DGLHeteroGraph, r:float, num_nearest_init_neighbors_per_type, pairs_per_level=10, device="cpu"):
        self.original_graph = graph
        self.summarized_graph = deepcopy(graph)
        self.summarized_graph = self.summarized_graph.to(device)
        
        self.r = r
        self.device = device
        self.num_nearest_init_neighbors_per_type = num_nearest_init_neighbors_per_type
        self.pairs_per_level = pairs_per_level
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            self.summarized_graph.nodes[src_type].data['node_size'] = torch.ones(self.summarized_graph.num_nodes(src_type), device=self.device)
            self.summarized_graph.edges[etype].data["adj"] = torch.ones(self.summarized_graph.num_edges(etype=etype), device=self.device)
        
        self._update_deg()
        
            
        
        total_num_nodes = graph.num_nodes()
        self.ntype_distribution = dict()
        for ntype in graph.ntypes:
            self.ntype_distribution[ntype] = graph.num_nodes(ntype) / total_num_nodes
        pass
    
    def _update_deg(self):
        rev_sub_g = dgl.reverse(self.summarized_graph,
                        copy_edata=True,      # duplicates every edge feature tensor
                        share_ndata=True)     # node features remain shared views
        for src_type, etype, dst_type in self.summarized_graph.canonical_etypes:
            
            
            rev_sub_g.update_all(fn.copy_e('adj', 'm'), fn.sum('m', f'deg_{etype}'), etype=etype)
            self.summarized_graph.nodes[src_type].data[f"deg_{etype}"] = rev_sub_g.nodes[src_type].data[f"deg_{etype}"]
    
    @abstractmethod
    def _create_gnn_layer(self):
        pass    
    
    def _get_adj(self, nodes_u, nodes_v, etype):
        exists = self.summarized_graph.has_edges_between(nodes_u, nodes_v, etype=etype)
        adj = torch.zeros(len(nodes_u), device=self.device)
        
                       # Get indices where the edge exists
        existing_edge_indices = exists.nonzero(as_tuple=True)[0]


        # Get edge IDs for the existing edges
        edge_ids = self.summarized_graph.edge_ids(nodes_u[existing_edge_indices], nodes_v[existing_edge_indices], etype=etype)
        #g_new.nodes[node_type].data[f'i{etype}'][new_nodes] =   infl_uv
    
        edge_data = self.summarized_graph.edges[etype].data['adj'][edge_ids]

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
                H = self.summarized_graph.nodes[node_type].data[f'h{etype}'].float()
      

            H = H.to(self.device)
            return H

        def _query( H):
            # compute pairwise L1 distances
            dist_matrix = torch.cdist(H, H, p=1)
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
        
        print("_init_costs", time.time() - start_time)
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
        print("_get_union", time.time() -start_time)
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
            k = min(self.num_nearest_init_neighbors_per_type[ntype] * self.pairs_per_level, costs.shape[0])
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

                topk_non_overlapping.append((src_node, dst_node))
                nodes.add(src_node)
                nodes.add(dst_node)

            topk_non_overlapping_per_type[ntype] = topk_non_overlapping

        print("_find_lowest_cost_edges", time.time() - start_time)
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
  
    
    def _merge_nodes(self, g_before):
            """
      
            """
            start_time = time.time()
            g_new = deepcopy(g_before)
            mappings = dict()
            for node_type, node_pairs in self.candidates.items():
                
                nodes_u = torch.tensor([i for i, _ in node_pairs], dtype=torch.int64, device=self.device)
                nodes_v = torch.tensor([i for  _,i in node_pairs], dtype=torch.int64, device=self.device)
 
                mapping = torch.arange(0, g_new.num_nodes(ntype=node_type), device=self.device )
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
                
                
                mapping[nodes_u] = g_new.nodes(ntype=node_type)[num_nodes_before:]
                mapping[nodes_v] = g_new.nodes(ntype=node_type)[num_nodes_before:]
                counts_u = (mapping.unsqueeze(1) > nodes_u).sum(dim=1) 
                counts_v = (mapping.unsqueeze(1) > nodes_v).sum(dim=1) 
                
                mapping = mapping - counts_u - counts_v
                mappings[node_type] = mapping
                
                feat_u = g_new.nodes[node_type].data["feat"][nodes_u]
                feat_v = g_new.nodes[node_type].data["feat"][nodes_v]
                feat_uv = (feat_u * cu + feat_v * cv) / (cu + cv)
                new_graph_size = g_new.num_nodes(ntype=node_type)
                g_new.nodes[node_type].data["feat_u"] = torch.zeros((new_graph_size,1), device=self.device)
                g_new.nodes[node_type].data["feat_u"][super_nodes] = feat_u
                g_new.nodes[node_type].data["feat_v"] = torch.zeros((new_graph_size,1), device=self.device)
                g_new.nodes[node_type].data["feat_v"][super_nodes] = feat_v
                g_new.nodes[node_type].data["feat"][super_nodes] = feat_uv
                
                
                cu = g_new.nodes[node_type].data["node_size"][nodes_u]
                cv = g_new.nodes[node_type].data["node_size"][nodes_v]
                
                g_new.nodes[node_type].data["node_size_u"] = torch.zeros(new_graph_size, device=self.device)
                g_new.nodes[node_type].data["node_size_u"][super_nodes] = cu
                g_new.nodes[node_type].data["node_size_v"] = torch.zeros(new_graph_size, device=self.device)
                
                g_new.nodes[node_type].data["node_size_v"][super_nodes] = cv
                cuv = cu + cv
                
                g_new.nodes[node_type].data["node_size"][super_nodes] = cuv
                    
                for src_type, etype,dst_type in g_new.canonical_etypes:
                    if src_type != node_type:
                        continue
                    
                    du = g_new.nodes[node_type].data[f"deg_{etype}"][nodes_u]
                    dv = g_new.nodes[node_type].data[f"deg_{etype}"][nodes_v]
                    duv = du + dv
                    g_new.nodes[node_type].data[f"deg_{etype}"][super_nodes] = duv
                    g_new.nodes[node_type].data[f"deg_{etype}_u"] = torch.zeros(new_graph_size, device=self.device)
                    g_new.nodes[node_type].data[f"deg_{etype}_u"][super_nodes] = du
                    g_new.nodes[node_type].data[f"deg_{etype}_v"] = torch.zeros(new_graph_size, device=self.device)
                    g_new.nodes[node_type].data[f"deg_{etype}_v"][super_nodes] = dv
                    su = g_new.nodes[node_type].data[f's{etype}'][nodes_u]  
                    sv =  g_new.nodes[node_type].data[f's{etype}'][nodes_v] 
                    
                    g_new.nodes[node_type].data[f's{etype}_u'] = torch.zeros((new_graph_size,1), device=self.device)
                    g_new.nodes[node_type].data[f's{etype}_u'][super_nodes] = su
                    g_new.nodes[node_type].data[f's{etype}_v'] = torch.zeros((new_graph_size,1), device=self.device)
                    g_new.nodes[node_type].data[f's{etype}_v'][super_nodes] = sv
                    
                    adj_uv = self._get_adj(nodes_u, nodes_v, etype)
                    adj_vu = self._get_adj(nodes_v, nodes_u, etype)
                    
                    
                    
                    suv = su - (adj_vu / (torch.sqrt(du + cu ))).unsqueeze(1) * feat_u  + sv -   (adj_uv / (torch.sqrt(dv + cv ))).unsqueeze(1)  * feat_v
                    g_new.nodes[node_type].data[f"s{etype}"][super_nodes] = suv
                    
                    mapping_u = torch.stack((nodes_u, super_nodes), dim=1) 
                    mapping_v = torch.stack((nodes_v, super_nodes), dim=1)
                    
                    sorted_indices_u = torch.argsort(mapping_u[:, 0])
                    sorted_mapping_u = mapping_u[sorted_indices_u]
                    
                    sorted_indices_v = torch.argsort(mapping_v[:, 0])
                    sorted_mapping_v = mapping_v[sorted_indices_v]
                    
                    src, dst, eid = g_new.out_edges(nodes_u, form="all", etype=etype)
                    indices = torch.searchsorted(sorted_mapping_u[:, 0], src)
                    result = sorted_mapping_u[indices, 1]
                    adj = g_new.edges[etype].data["adj"][eid]
                    t = torch.stack((src,dst), dim=1)
                    edges_from_u_super = torch.stack((result,dst), dim=1)
                    g_new.add_edges( edges_from_u_super[:,0], edges_from_u_super[:,1],data={"adj": adj}, etype=etype)
                    
                    
                    
                    src, dst , eid = g_new.out_edges(nodes_v, form="all", etype=etype)
                    indices = torch.searchsorted(sorted_mapping_v[:, 0], src)
                    result = sorted_mapping_v[indices, 1]
                    adj = g_new.edges[etype].data["adj"][eid]
                    edges_from_u_super = torch.stack((result,dst), dim=1)
                    g_new.add_edges( edges_from_u_super[:,0], edges_from_u_super[:,1],data={"adj": adj}, etype=etype)
                     
                    
                    src, dst, eid = g_new.in_edges(nodes_u, form="all", etype=etype)
                   
                    # Now use searchsorted on the sorted mapping
                    indices = torch.searchsorted(sorted_mapping_u[:, 0], dst)
                    result = sorted_mapping_u[indices, 1]
                    adj = g_new.edges[etype].data["adj"][eid]
                    edges_from_u_super = torch.stack((src,result), dim=1)
                    g_new.add_edges( edges_from_u_super[:,0], edges_from_u_super[:,1],data={"adj": adj}, etype=etype)
                    
                    
                    src, dst, eid = g_new.in_edges(nodes_v, form="all", etype=etype)
                    
                    indices = torch.searchsorted(sorted_mapping_v[:, 0], dst)
                    result = sorted_mapping_v[indices, 1]
                    adj = g_new.edges[etype].data["adj"][eid]
                    edges_from_u_super = torch.stack((src,result), dim=1)
                    
                   # edges_from_u_super = edges_from_u_super[mask]
                    g_new.add_edges( edges_from_u_super[:,0], edges_from_u_super[:,1],data={"adj": adj}, etype=etype)
                    
                    
                    nodes_uv = torch.cat([nodes_u, nodes_v])
                    
                    edges = g_new.edges()
                    edge_pairs = torch.stack((edges[0], edges[1]), dim=1)
                    mask = torch.logical_and(self.vectorwise_isin(edge_pairs, mapping_u) == False , self.vectorwise_isin(edge_pairs, mapping_v) == False)
                    mask = mask == False
                    ids = g_new.edge_ids(edges[0][mask], edges[1][mask])
                    g_new.remove_edges(ids)
                   
                    edges_src, edges_dst = g_new.in_edges(nodes_uv,  etype=etype)
                    
                    
                    
                    
                    def msg_minus_neigh_s(edges):
                        return {'min':  (edges.data['adj'].unsqueeze(1) *edges.src["feat"])/torch.sqrt(edges.src[f"deg_{etype}"] + edges.src['node_size']).unsqueeze(1)  } #

                    def reduce_minus_neigh_s(nodes):
                        return {f's_new': torch.sum(nodes.mailbox['min']  , dim=1)}
                    rev_sub_g = dgl.reverse(g_new,
                        copy_edata=True,      # duplicates every edge feature tensor
                        share_ndata=True)     # node features remain shared views
                    rev_sub_g.send_and_recv((edges_dst,edges_src ), message_func=msg_minus_neigh_s, reduce_func=reduce_minus_neigh_s, etype=etype)
                    g_new.nodes[src_type].data[f"s{etype}"] -= rev_sub_g.nodes[src_type].data["s_new"]
                    
                    
                    edges_src, edges_dst = g_new.in_edges(super_nodes,  etype=etype)

                    rev_sub_g = dgl.reverse(g_new,
                        copy_edata=True,      # duplicates every edge feature tensor
                        share_ndata=True)     # node features remain shared views
                    rev_sub_g.send_and_recv((edges_dst,edges_src ), message_func=msg_minus_neigh_s, reduce_func=reduce_minus_neigh_s, etype=etype)
                    g_new.nodes[src_type].data[f"s{etype}"] += rev_sub_g.nodes[src_type].data["s_new"]
                    
                    
                    
                    

                nodes_to_delete = torch.cat([nodes_u, nodes_v])           
                g_new.remove_nodes(nodes_to_delete, ntype=node_type)
                
                for src_type, etype,dst_type in g_new.canonical_etypes:
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

            
                    if src_type == dst_type:
                        g_new = dgl.remove_self_loop(g_new, etype=etype)
                    
                    
                    
                            
                    
            print("_merge_nodes", time.time() - start_time)
            return g_new, mappings

                
    def _update_s(self, g_before, g_after, mappings):
        for src_type, etype,dst_type in g_before.canonical_etypes:
            mapping_src = mappings[src_type]
            mapping_dst = mappings[dst_type]

            


    def _calc_costs(self):
        pass
    

    
    def init(self):
        self._create_gnn_layer()
        init_costs = self._init_costs()
        type_pairs = self._get_union(init_costs)
        self._init_merge_graphs(type_pairs)
        self.candidates = {"user": [(1,3),(2,4)]} #self._find_lowest_costs()
       
    def coarsen_step(self):
        
        new_g, mappings = self._merge_nodes(self.summarized_graph)
        
        t = 2
    