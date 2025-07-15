import dgl
import torch
import numpy as np
from Coarsener.HeteroRGCNCoarsener import HeteroRGCNCoarsener
from Data.Citeseer import Citeseer
from Data.DBLP import DBLP
from Data.IMDB import IMDB
from Data.ACM import ACM
from Data.Cora import Cora
#from Data.Actor import Actor
from Projections.ContrastiveLearner import NonLinearContrastiveLearner, LinearContrastiveLearner
from Projections.CCA import CCA
from Projections.DeepCCA import DeepCCA

from Models.SimpleHeteroGCN import HeteroGCNCiteer    
from Models.ImprovedGCN import ImprovedGCN
from Models.HeteroSage import HeteroSAGE
from Models.HeteroSGC import HeteroSGCPaper

from Experiments.model_helper import run_experiments
from Projections.AutoEncoder import MultiviewAutoencoder
import os
import pandas as pd
def get_proj(name):
    if name == 'CCA':
        return CCA
    elif name == 'CLL':
        return LinearContrastiveLearner
    elif name == 'CLNL':
        return NonLinearContrastiveLearner
    elif name == "AUTO":
        return MultiviewAutoencoder
    elif name == 'DeepCCA':
        return DeepCCA
    else:
        return None

def coarsen_graph(dataset,proj_name=None, pairs_per_level=5, device="cuda:0", n_components=64, zscore=False, num_neighbors_per_ntype=40,num_neighbors_per_etype=40, checkpoints=None, batch_size=None, run=0):
    # Make sure we can use CUDA
    try: 
        torch.cuda.empty_cache()

        # Default neighbor configuration if not provided
        
        
        
        # Default checkpoints if not provided
        if checkpoints is None:
            checkpoints = [0.9999,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,0.02,0.01,0.005]
        
        # Load the dataset and graph
        
        

        original_graph = dataset.load_graph(n_components=n_components)
        original_graph = original_graph.to(device)
        
        num_neighbors = {}
        for ntype in original_graph.ntypes:
            num_neighbors[ntype] = num_neighbors_per_ntype
        for _,etype, _ in original_graph.canonical_etypes:
            num_neighbors[etype] = num_neighbors_per_etype
        # Initialize the coarsener
        
        proj = None
        if proj_name != None:
            proj = get_proj(proj_name)

        pca_name = ""
        if n_components != None:
            pca_name = f'pca_{n_components}'
        folder_name = f'{type(dataset).__name__}_{proj_name}_{pca_name}_5_pairs_{run}' 
        coarsener = HeteroRGCNCoarsener(
            original_graph, 
            num_nearest_init_neighbors_per_type=num_neighbors, 
            use_zscore=zscore,
            device=device,
            cca_cls=proj,
            checkpoints=checkpoints,
            folder_name=folder_name,
            batch_size=batch_size,
            pairs_per_level=pairs_per_level,
            norm_p=1,
            approx_neigh=True,
            add_feat=not zscore,
            use_out_degree=False
        )

        # Initialize and summarize the coarsener
        coarsener.init()
        coarsener.summarize()

        return coarsener
    except Exception as e:
        print('error!' , e)



# for i in range(5):
   
d = Cora()

coarsen_graph(d, proj_name="CLNL", n_components=64, run=0)
coarsen_graph(d, proj_name="CLL", n_components=30   , run=0)
coarsen_graph(d, proj_name="AUTO", n_components=30, run=0)
coarsen_graph(d, proj_name="CCA", n_components=30, run=0)



#nohup python run.py > output.log 2>&1 &