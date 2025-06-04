import unittest
import torch
import dgl

from Coarsener.HeteroRGCNCoarsener import HeteroRGCNCoarsener
class SingleTester(unittest.TestCase):
    def setUp(self):
        
        self.homo_graph =  dgl.heterograph({
            ('user', 'follows', 'user'): ([0, 1, 1, 1, 2], [1, 2, 3, 4,3])})
        self.homo_graph.nodes['user'].data['feat'] = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0]])
        num_nearest_init_neighbors_per_type = {"follows": 3, "user": 2}
        self.coarsener = HeteroRGCNCoarsener(self.homo_graph, 0.4, num_nearest_init_neighbors_per_type, device="cpu", approx_neigh= False, add_feat=True)
        
        self.device = self.homo_graph.device
        print("Starting rgcn test...")

    def tearDown(self):
        print("rgcn test finished.")
    
    def test_r_gcn_initialization(self):
        
        self.coarsener._create_gnn_layer()
        
        s = torch.tensor([
            1,
            11.12,
            4,
            0,
            0
        ], device = self.device)
        h = torch.tensor( [
            1.207,
            6.06,
            4.33,
            4,
            5
            
        ], device=self.device)
        
        i = torch.tensor( [
            0.5,
            2.71,
            1,
            0,
            0    
        ],device= self.device)
        
        
        
        torch.testing.assert_close(self.coarsener.summarized_graph.nodes["user"].data[f'sfollows'], s.unsqueeze(1), rtol=0, atol=0.1)  
        torch.testing.assert_close(self.coarsener.summarized_graph.nodes["user"].data[f'hfollows'], h.unsqueeze(1), rtol=0, atol=0.1)  
     #   torch.testing.assert_close(self.coarsener.summarized_graph.nodes["user"].data[f'ifollows'], i, rtol=0, atol=0.1)  
        

    
    
    
    def test_merge(self):
        pass
    
    def test_create_H_merged(self):
        self.coarsener._create_gnn_layer()
        H_merged = self.coarsener._create_h_merged([0,0,0], [1,2,3], ntype="user", etype="follows")
        H = torch.tensor( [
            5.29,
            3.5,
            2.24   
        ],device= self.device)
        
        torch.testing.assert_close(H_merged, H.unsqueeze(1), rtol=0, atol=0.1)
        
        
    
    def test_neighbors_costs(self):
        self.coarsener._create_gnn_layer()
        
        h_node = self.coarsener.summarized_graph.nodes["user"].data[f"hfollows"]
        d_node = self.coarsener.summarized_graph.nodes["user"].data[f"deg_follows"]
        c_node = cv = self.coarsener.summarized_graph.nodes["user"].data["node_size"]
        s_node =  self.coarsener.summarized_graph.nodes["user"].data[f"sfollows"]
        feat_node = self.coarsener.summarized_graph.nodes["user"].data[f"feat"]
        infl_node = self.coarsener.summarized_graph.nodes["user"].data[f"ifollows"]
        
        
        pass
        
    def test_neighbors_costs_approx(self):
        self.homo_graph =  dgl.heterograph({
            ('user', 'follows', 'user'): ([0, 1, 2], [1, 2, 3])})
        self.homo_graph.nodes['user'].data['feat'] = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
        num_nearest_init_neighbors_per_type = {"follows": 3, "user": 2}
        self.coarsener = HeteroRGCNCoarsener(self.homo_graph, 0.4, num_nearest_init_neighbors_per_type, device="cpu", approx_neigh= False)
        self.device = self.homo_graph.device
        self.coarsener._create_gnn_layer()
        #self.coarsener._init_merge_graphs({"user": torch.tensor([[0,3]])})
        pairs = torch.tensor([[0, 3]], device= self.device)
        h_node = self.coarsener.summarized_graph.nodes["user"].data[f"hfollows"]
        d_node = self.coarsener.summarized_graph.nodes["user"].data[f"deg_follows"]
        c_node = cv = self.coarsener.summarized_graph.nodes["user"].data["node_size"]
        s_node =  self.coarsener.summarized_graph.nodes["user"].data[f"sfollows"]
        feat_node = self.coarsener.summarized_graph.nodes["user"].data[f"feat"]
        infl_node = self.coarsener.summarized_graph.nodes["user"].data[f"ifollows"]
        
        costs = self.coarsener.neigbor_approx_difference_per_pair(self.homo_graph, pairs,  d_node, c_node, infl_node, feat_node, "follows")
        torch.testing.assert_close(costs, torch.tensor([1.81], device=self.device), rtol=0, atol=0.1)
        
        
        pass
    

    def test_total_costs(self):
        self.coarsener._create_gnn_layer()
        self.coarsener._init_merge_graphs({"user": torch.tensor([[0,1] ,[0,2],[0,3]])})
        total_costs = torch.tensor([
            4.853,
            3.68,
            5.883000000000001
        ], device=self.device)
        torch.testing.assert_close(self.coarsener.merge_graphs["user"].edata["costs"], total_costs, rtol=0, atol=0.1)
        
    
    def test_merge_step(self):
        self.coarsener._create_gnn_layer()
        #self.coarsener._init_merge_graphs({"user": torch.tensor([[0,1] ,[0,2],[0,3]])})
        self.coarsener.candidates = {"user": torch.tensor([[2, 4], [1, 3]])}
        merged_s_real = torch.tensor(
            [[1.3416],
             [1.3416],
             [5.9604]], device=self.device)
        
        
        merged_s_real = torch.tensor(
            [[1.3416],
             [1.3416],
             [5.9604]], device=self.device)
        
        
        g, mapping = self.coarsener._merge_nodes(self.coarsener.summarized_graph)
        mapping_real = torch.tensor([0,2,1,2,1], device=self.device)
        
        torch.testing.assert_close(mapping_real, mapping["user"], rtol=0, atol=0.1)
        torch.testing.assert_close(g.nodes["user"].data[f"sfollows"], merged_s_real, rtol=0, atol=0.1)
      