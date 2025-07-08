import dgl
import torch
import numpy as np
from Coarsener.HeteroRGCNCoarsener import HeteroRGCNCoarsener
from Data.Citeseer import Citeseer
from Data.DBLP import DBLP
from Data.IMDB import IMDB
from Data.ACM import ACM
from Data.Actor import Actor
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

def coarsen_graph(dataset,proj_name=None, pairs_per_level=20, device="cuda:0", n_components=30, num_neighbors_per_ntype=25,num_neighbors_per_etype=25, checkpoints=None, batch_size=None):
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
        folder_name = f'{type(dataset).__name__}_{proj_name}_{pca_name}' 
        coarsener = HeteroRGCNCoarsener(
            original_graph, 
            num_nearest_init_neighbors_per_type=num_neighbors, 
            use_zscore=False,
            device=device,
            cca_cls=proj,
            checkpoints=checkpoints,
            folder_name=folder_name,
            batch_size=batch_size,
            pairs_per_level=pairs_per_level,
            norm_p=1,
            approx_neigh=True,
            add_feat=True,
            use_out_degree=False
        )

        # Initialize and summarize the coarsener
        coarsener.init()
        coarsener.summarize()

        return coarsener
    except Exception as e:
        print(e)


import os

def get_all_files(root_folder):
    file_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    return file_paths


def get_node_type(name):
    if 'ACM' in name:
        return "paper"
    elif 'DBLP' in name:
        return 'author'
    elif "Cit" in name or 'cit' in name:
        return 'paper'
    elif 'ACTOR' in name:
        return 'paper'
    


def eval( model=HeteroSGCPaper ): 
    files = get_all_files('results/')
    print(files)
    for f in files:
        try: 
            import pickle
        
            device= "cuda:0"
            
            with open(f, 'rb') as fh:
                    
                coarsener = pickle.load(fh) 
            node_target_type = get_node_type(f)
            print(f)
            print(node_target_type)
            original_graph = coarsener.original_graph
            coarsend_graph = coarsener.summarized_graph

            #coarsend_graph = coarsend_graph.cpu()
            mapping = coarsener.get_mapping(node_target_type)
            coarsener.make_mask(mapping, node_target_type)

            labels = coarsener.get_labels(mapping, node_target_type)
            coarsend_graph.nodes[node_target_type].data["label"] = torch.tensor([labels[i] for i in range(len(labels)) ],  device=coarsend_graph.device) #,
            print("ratio", coarsend_graph.num_nodes()/ original_graph.num_nodes() ) 


            orig, coar, loss_ori, loss_coar ,_,_= run_experiments(original_graph, coarsend_graph,  model,
                                                            model_param={"hidden_dim": 64,"num_layers":4},
                                    optimizer_param={"lr": 0.01, "weight_decay": 5e-4},
                                    num_runs=1, epochs=400,eval_interval=1, target_node_type=node_target_type)
            orig_short = [ o[-1] for o in orig ]
            coar_short = [ o[-1] for o in coar ]
            print(f)
            print(max(orig_short), max(coar_short))

        except Exception as e:
            print(e)


eval()

# d = IMDB()

# coarsen_graph(d, proj_name="CCA")
# coarsen_graph(d, proj_name="AUTO")
# coarsen_graph(d, proj_name="CLL")
# coarsen_graph(d, proj_name="CLNL")
# d = ACM()

# coarsen_graph(d, proj_name="CCA")
# coarsen_graph(d, proj_name="AUTO")
# coarsen_graph(d, proj_name="CLL")
# coarsen_graph(d, proj_name="CLNL")

# d = DBLP()

# coarsen_graph(d, proj_name="CCA")
# coarsen_graph(d, proj_name="AUTO")
# coarsen_graph(d, proj_name="CLL")
# coarsen_graph(d, proj_name="CLNL")

# d = Citeseer()

# coarsen_graph(d)
# coarsen_graph(d, proj_name="AUTO")
# coarsen_graph(d, proj_name="CLL")
# coarsen_graph(d, proj_name="CLNL")
# d = Actor()
# coarsen_graph(d )
# coarsen_graph(d, proj_name="CCA")
# coarsen_graph(d, proj_name="AUTO")
# coarsen_graph(d, proj_name="CLL")
# coarsen_graph(d, proj_name="CLNL")

# d = DBLP()

# coarsen_graph(d, proj_name="CCA")
# coarsen_graph(d, proj_name="AUTO")
# coarsen_graph(d, proj_name="CLL")
# coarsen_graph(d, proj_name="CLNL")

#nohup python run.py > output.log 2>&1 &