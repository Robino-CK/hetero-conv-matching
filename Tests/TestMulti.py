import unittest
import torch
import dgl
import numpy as np
from Coarsener.HeteroRGCNCoarsener import HeteroRGCNCoarsener
class MultiGraphTest(unittest.TestCase):
    def setUp(self):
        
        self.graph =  dgl.heterograph({
            ('square', 'squre_to_cicle', 'circle'): ([0, 1, 1], [0, 0, 1]),
            ('circle', 'circle_to_triangle', 'triangle'): ([0, 0,1,1], [0,1,1,2])})
            
            
            
        self.graph.nodes['square'].data['feat'] = torch.zeros(self.graph.num_nodes(ntype="square"))
        self.graph.nodes['circle'].data['feat'] = torch.tensor( [[2.], [4.]])
        
        self.graph.nodes['triangle'].data['feat'] = torch.tensor( [[1.], [2.],[3.]])
        num_nearest_init_neighbors_per_type = {"square": 3, "squre_to_cicle": 2, "circle": 2, "circle_to_triangle": 2, "triangle": 2}
        self.coarsener = HeteroRGCNCoarsener(self.graph, 0.4, num_nearest_init_neighbors_per_type, device="cpu", add_feat=False, approx_neigh= False)
        
        self.device = self.graph.device
        print("Starting rgcn test...")

    def tearDown(self):
        print("rgcn test finished.")
    
    def test_r_gcn_initialization(self):
        
        self.coarsener._create_gnn_layer()
        
        s = torch.tensor([
            1 / torch.sqrt(torch.tensor(2)) + 2 / torch.sqrt(torch.tensor(3)),
            2 / torch.sqrt(torch.tensor(3)) + 3 / torch.sqrt(torch.tensor(2)),
            
        ], device = self.device)
        h = torch.tensor( [
            s[0] / 1 / torch.sqrt(torch.tensor(3)),
            s[1] / 1 / torch.sqrt(torch.tensor(3)),
            
            
        ], device=self.device)
        
        i = torch.tensor( [
            0.5,
            2.71,
            1,
            0,
            0    
        ],device= self.device)
        
        
        
        torch.testing.assert_close(self.coarsener.summarized_graph.nodes["circle"].data[f'scircle_to_triangle'], s.unsqueeze(1), rtol=0, atol=0.1)  
        torch.testing.assert_close(self.coarsener.summarized_graph.nodes["circle"].data[f'hcircle_to_triangle'], h.unsqueeze(1), rtol=0, atol=0.1)  
    
    def test_merge(self):
        pass
    
    def test_create_H_merged(self):
        
        self.coarsener._create_gnn_layer()
        H_merged = self.coarsener._create_h_merged([0], [1], ntype="circle", etype="circle_to_triangle")
        H = torch.tensor( [
            (torch.sqrt(torch.tensor(3)) + torch.sqrt(torch.tensor(2) ) ) * ( 2 / 3)
        ],device= self.device)
        
        torch.testing.assert_close(H_merged, H.unsqueeze(1), rtol=0, atol=0.1)
    
    def test_merge_step(self):
        self.coarsener._create_gnn_layer()
        #self.coarsener._init_merge_graphs({"user": torch.tensor([[0,1] ,[0,2],[0,3]])})
        self.coarsener.candidates = {"square": [],  "circle": torch.tensor([[0, 1] ], dtype=torch.int64), "triangle" : []} #
        
        merged_h_real = torch.tensor(
            [[5.1378 / torch.sqrt(torch.tensor(6))]], device=self.device)
        
        
        g, mapping = self.coarsener._merge_nodes(self.coarsener.summarized_graph)
        torch.testing.assert_close(g.nodes["circle"].data[f"hcircle_to_triangle"], merged_h_real, rtol=0, atol=0.1)
    
        self.coarsener.candidates = {"square": [],   "triangle" : torch.tensor([[0,2]], dtype=torch.int64), "circle": torch.tensor([[0, 1] ], dtype=torch.int64), } #"triangle" : []
        
        
        merged_h_real = torch.tensor(
            [[ (4 / torch.sqrt(torch.tensor(3 )) + 2) / torch.sqrt(torch.tensor(6))]], device=self.device)
        
        
        g, mapping = self.coarsener._merge_nodes(self.coarsener.summarized_graph)
        torch.testing.assert_close(g.nodes["circle"].data[f"hcircle_to_triangle"], merged_h_real, rtol=0, atol=0.1)
        pass