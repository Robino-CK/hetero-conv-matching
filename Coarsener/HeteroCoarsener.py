import torch
import dgl
from copy import deepcopy
from abc import ABC, abstractmethod
import time


class HeteroCoarsener(ABC):
    def __init__(self, graph: dgl.DGLHeteroGraph, r:float, device="cpu"):
        self.original_graph = graph
        self.summarized_graph = deepcopy(graph)
        self.summarized_graph = self.summarized_graph.to(device)
        self.r = r
        self.device = device
        pass
    
    @abstractmethod
    def _create_gnn_layer(self):
        pass    
    
    
    
    
    def _init_merge_graphs(self):
        
        
        pass

    
    def _init_costs(self):
        def _compute_embeddings(node_type, etype=None, use_feat=True):
            # Select the proper feature tensor
            if use_feat and 'feat' in self.coarsened_graph.nodes[node_type].data:
                H = self.coarsened_graph.nodes[node_type].data['feat'].float()
            elif etype is not None and f'h{etype}' in self.coarsened_graph.nodes[node_type].data:
                H = self.coarsened_graph.nodes[node_type].data[f'h{etype}'].float()
            else:
                # TODO this case should not happen
                pass

            H = H.to(self.device)
            return H

        def _query( H):
            # compute pairwise L1 distances
            dist_matrix = torch.cdist(H, H, p=1)
            # get k smallest distances (including self)
            k = min(self.num_nearest_per_etype, H.size(0))
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
        for src_type, etype, dst_type in self.coarsened_graph.canonical_etypes:
           init_costs[etype] = _init_calculate_clostest_edges(src_type, etype, dst_type)
          
        for ntype in self.coarsened_graph.ntypes:
            if "feat" in self.coarsened_graph.nodes[dst_type].data:
                init_costs[ntype] = _init_calculate_clostest_features(ntype)   
        
        print("_init_costs", time.time() - start_time)
        return init_costs
    
    def _get_union(self, init_costs):
        start_time = time.time()
        closest = {src: {} for src,_,_ in self.coarsened_graph.canonical_etypes}
        # edges
        for src, etype, _ in self.coarsened_graph.canonical_etypes:
            dists, idxs = init_costs[etype]
            for i, neighbors in enumerate(idxs):
                neigh = set(neighbors.tolist()) - {i}
                closest[src].setdefault(i, set()).update(neigh)
        # features
        for ntype in self.coarsened_graph.ntypes:
            if not ntype in init_costs.keys():
                continue 
            dists, idxs = init_costs[ntype]
            for i, neighbors in enumerate(idxs):
                neigh = set(neighbors.tolist()) - {i}
                closest.setdefault(ntype, {})
                closest[ntype].setdefault(i, set()).update(neigh)
        print("_get_union", time.time() -start_time)
        return closest 
    
    def init(self):
        self._create_gnn_layer()
        init_costs = self._init_costs()
        self.init_neighbors = self._get_union(init_costs)
        
    
    def _merge_nodes(self):
        pass

    def _calc_costs(self):
        pass
    
    def _find_lowest_costs(self):
        pass
    
    