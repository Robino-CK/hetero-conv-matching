
from Coarsener.HeteroRGCNCoarsener import HeteroRGCNCoarsener
from Data.test_graphs import TestHetero, TestHomo

if __name__ == "__main__":
    test = TestHomo()
    num_nearest_init_neighbors_per_type = {"follows": 10, "user": 10}
    coarsener = HeteroRGCNCoarsener(test.g, 0.4, num_nearest_init_neighbors_per_type)
    coarsener.init()