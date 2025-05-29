
from Coarsener.HeteroRGCNCoarsener import HeteroRGCNCoarsener
from Data.test_graphs import TestHetero, TestHomo

if __name__ == "__main__":
    test = TestHomo()
    coarsener = HeteroRGCNCoarsener(test.g, 0.4)
    coarsener.init()