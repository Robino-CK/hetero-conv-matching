
import torch
print("hi")
from Coarsener.HeteroRGCNCoarsener import HeteroRGCNCoarsener

from Data.Citeseer import Citeseer
import dgl.function as fn
from Tests.TestSingle import SingleTester
from Tests.TestMulti import MultiGraphTest
import unittest
from Data.DBLP import DBLP
import torch
if __name__ == "__main__":
    
    dataset = Citeseer() 
    original_graph = dataset.load_graph(n_components=10)
    num_nearest_init_neighbors_per_type = {"papertoauthor": 3, "authortopaper": 3, "conferencetopaper":3, "papertoconference":3,"papertoterm":3, "termtopaper":3 }

    coarsener = HeteroRGCNCoarsener(original_graph, 0.4, num_nearest_init_neighbors_per_type, device="cpu", pairs_per_level=10,norm_p=2, approx_neigh=False, add_feat=False, use_out_degree=False) 

    coarsener.init()
    
    
    # dataset = Citeseer() 
    # original_graph = dataset.load_graph()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # original_graph = original_graph.to(device=device)
    # print(device)
    
      
    # num_nearest_init_neighbors_per_type = {"paper": 1, "cites": 3, "cited-by":1}
    # coarsener = HeteroRGCNCoarsener(original_graph, 0.4, num_nearest_init_neighbors_per_type, device=device, pairs_per_level=10,norm_p=2, approx_neigh=False) 
    # coarsener.init()
    
    # coarsener.coarsen_step()
    
    
    # coarsener = HeteroRGCNCoarsener(original_graph, 0.4, num_nearest_init_neighbors_per_type, device=device, pairs_per_level=10,norm_p=2, add_feat=False, approx_neigh=False) 
    # coarsener.init()
    # coarsener.coarsen_step()
    
    single_tester = SingleTester()
    single_tester.setUp()
    single_tester.test_total_costs()
    
    # multi_tester = MultiGraphTest()
    # multi_tester.setUp()
    # multi_tester.test_merge_step()
    # suite = unittest.TestLoader().loadTestsFromTestCase(MultiGraphTest)
    # unittest.TextTestRunner().run(suite)
    
    
    suite = unittest.TestLoader().loadTestsFromTestCase(SingleTester)
    unittest.TextTestRunner().run(suite)
    
    # coarsener.coarsen_step()
    # cit = Citeseer()
    # g = cit.load_graph()
    # num_nearest_init_neighbors_per_type = {"paper": 20, "cites": 20, "cited-by":20}
    # coarsener = HeteroRGCNCoarsener(cit.load_graph(), 0.4, num_nearest_init_neighbors_per_type)
    # print("hi")
    # coarsener.init()
    
    # for i in range(300):
    #     print("HELP!!!!")
    #     #cProfile.run('re.compile("foo|bar")')
    #     coarsener.coarsen_step()
        
    # mapping = coarsener.get_mapping("paper")    
    # print(coarsener.summarized_graph)