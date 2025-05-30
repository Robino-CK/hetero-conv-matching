
from Coarsener.HeteroRGCNCoarsener import HeteroRGCNCoarsener
from Data.test_graphs import TestHetero, TestHomo
from Data.Citeseer import Citeseer
import dgl.function as fn
if __name__ == "__main__":
    test = TestHomo()
    num_nearest_init_neighbors_per_type = {"follows": 10, "user": 10}
    coarsener = HeteroRGCNCoarsener(test.g, 0.4, num_nearest_init_neighbors_per_type)
    
    coarsener.init()
    coarsener.coarsen_step()
    cit = Citeseer()
    cit.load_graph()
   # num_nearest_init_neighbors_per_type = {"paper": 10, "cites": 10, "cited-by":10}
   # coarsener = HeteroRGCNCoarsener(cit.load_graph(), 0.4, num_nearest_init_neighbors_per_type)
    
   # coarsener.init()
   # coarsener.coarsen_step()