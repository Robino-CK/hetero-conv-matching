import unittest
import torch
import dgl
import numpy as np
from Coarsener.HeteroRGCNCoarsener import HeteroRGCNCoarsener

# src_all_orig = original_graph.edges(etype="authortopaper")[0]
# dst_all_orig = original_graph.edges(etype="authortopaper")[1]
# for src_orig, dst_orig in zip(src_all_orig, dst_all_orig):
#     src_c = mapping_author_array[src_orig.item()]
#     dst_c = mapping_paper_array[dst_orig.item()]
#     edges = coarsend_graph.edge_ids(src_c, dst_c, etype="authortopaper")
#     assert len(edges) != 0, "warning"
    
class MultiGraphTest(unittest.TestCase):
    def setUp(self):
        
        self.graph =  dgl.heterograph({
            ('square', 'square_to_circle', 'circle'): ([0, 1, 1], [0, 0, 1]),
            ('circle', 'circle_to_triangle', 'triangle'): ([0, 0,1,1], [0,1,1,2])})
            
            
            
        self.graph.nodes['square'].data['feat'] = torch.zeros(self.graph.num_nodes(ntype="square"))
        self.graph.nodes['circle'].data['feat'] = torch.tensor( [[2.], [4.]])
        
        self.graph.nodes['triangle'].data['feat'] = torch.tensor( [[1.], [2.],[3.]])
        num_nearest_init_neighbors_per_type = {"square": 3, "square_to_circle": 2, "circle": 2, "circle_to_triangle": 2, "triangle": 2}
        self.coarsener = HeteroRGCNCoarsener(self.graph, 0.4, num_nearest_init_neighbors_per_type, device="cpu", add_feat=False, approx_neigh=True, use_out_degree=False, use_zscore=False)
        
        self.device = self.graph.device
        print("Starting rgcn test...")

    def tearDown(self):
        print("rgcn test finished.")
        
    def test_r_gcn_i_init(self):
        self.coarsener.approx_neigh = True
        self.coarsener._create_gnn_layer()
        
    
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
        
        
        
        
        torch.testing.assert_close(self.coarsener.summarized_graph.nodes["circle"].data[f'scircle_to_triangle'], s.unsqueeze(1), rtol=0, atol=0.1)  
        torch.testing.assert_close(self.coarsener.summarized_graph.nodes["circle"].data[f'hcircle_to_triangle'], h.unsqueeze(1), rtol=0, atol=0.1)  
    
        
    
    
    
    def test_costs(self):
        self.coarsener.use_out_degree = False
        self.coarsener._create_gnn_layer()
        self.coarsener._init_merge_graphs({"circle" : torch.tensor([[0,1]] , device=self.device)})
        torch.testing.assert_close(self.coarsener.merge_graphs["circle"].edata["costs"], torch.tensor([2.78], device=self.device), rtol=0, atol=0.1 )
        pass
    
    def test_create_H_merged(self):
        
        self.coarsener._create_gnn_layer()
        H_merged = self.coarsener._create_h_merged([0], [1], ntype="circle", etype="circle_to_triangle")
        H = torch.tensor( [
            (torch.sqrt(torch.tensor(3)) + torch.sqrt(torch.tensor(2) ) ) * ( 2 / 3)
        ],device= self.device)
        
        torch.testing.assert_close(H_merged, H.unsqueeze(1), rtol=0, atol=0.1)
        
    def test_cca(self):
        self.coarsener.use_cca = True
        self.coarsener._create_gnn_layer()
        self.coarsener.init()
        t = 2
    
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
        deg_circle_to_triangle = torch.tensor(
            [4.], device=self.device
        )
        deg_circle_from_square = torch.tensor(
            [3.], device=self.device
        )
        deg_triangle = torch.tensor([2., 2.])
        deg_square = torch.tensor(
            [1., 2.], device=self.device
        ) 
        g, mapping = self.coarsener._merge_nodes(self.coarsener.summarized_graph)
        torch.testing.assert_close(g.nodes["circle"].data[f"deg_circle_to_triangle"], deg_circle_to_triangle, rtol=0, atol=0.1)
        torch.testing.assert_close(g.nodes["triangle"].data[f"deg_circle_to_triangle"], deg_triangle, rtol=0, atol=0.1)
        torch.testing.assert_close(g.nodes["circle"].data[f"deg_square_to_circle"], deg_circle_from_square, rtol=0, atol=0.1)
        torch.testing.assert_close(g.nodes["square"].data[f"deg_square_to_circle"], deg_square, rtol=0, atol=0.1)
        
        
        torch.testing.assert_close(g.nodes["circle"].data[f"hcircle_to_triangle"], merged_h_real, rtol=0, atol=0.1)
        pass