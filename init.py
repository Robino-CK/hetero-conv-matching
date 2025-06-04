
from Coarsener.HeteroRGCNCoarsener import HeteroRGCNCoarsener

from Data.Citeseer import Citeseer
import dgl.function as fn
from Tests.TestSingle import SingleTester
from Tests.TestMulti import MultiGraphTest
import unittest

if __name__ == "__main__":
    
    # single_tester = SingleTester()
    # single_tester.setUp()
    # single_tester.test_merge_step()
    
    # multi_tester = MultiGraphTest()
    # multi_tester.setUp()
    # multi_tester.test_merge_step()
    suite = unittest.TestLoader().loadTestsFromTestCase(MultiGraphTest)
    unittest.TextTestRunner().run(suite)
    
    
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