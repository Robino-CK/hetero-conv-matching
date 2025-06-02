
from Coarsener.HeteroRGCNCoarsener import HeteroRGCNCoarsener
from Data.test_graphs import TestHetero, TestHomo
from Data.Citeseer import Citeseer
import dgl.function as fn
from Tests.RGCNTest import RGCNTest
import unittest

if __name__ == "__main__":
    test = TestHomo()
    num_nearest_init_neighbors_per_type = {"follows": 3, "user": 2}
    coarsener = HeteroRGCNCoarsener(test.g, 0.4, num_nearest_init_neighbors_per_type, device="cpu", approx_neigh=True)
  
    coarsener.init()
    
    suite = unittest.TestLoader().loadTestsFromTestCase(RGCNTest)
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