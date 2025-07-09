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
import pandas as pd

import os

def get_all_files(root_folder):
    file_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    return file_paths


def get_node_type(name):
    if 'Actor' in name:
        return 'paper'
    elif 'IMDB' in name:
        return "movie"
    if 'ACM' in name:
        return "paper"
    elif 'DBLP' in name:
        return 'author'
    elif "Cit" in name or 'cit' in name:
        return 'paper'
    elif 'Actor' in name:
        return 'paper'
    else:
        return None
    
import pandas as pd


def update_row_by_ratio(df, columns, ratio, column_name, value):
    # Check if row with given ratio exists
    if ratio in df['ratio'].values:
        # Update the existing row
        df.loc[df['ratio'] == ratio, column_name] = value
    else:
        # Create a new row with NaNs and set ratio and column value
        new_row = pd.Series({col: None for col in columns})
        new_row['ratio'] = ratio
        new_row[column_name] = value
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    return df



def eval( model=HeteroSGCPaper ): 
    files = get_all_files('results/')
    print(files)
    columns = set()
    columns.add('ratio')
    for f in files:
        columns.add(f.split('/')[1])
        
    df = pd.DataFrame(columns=list(columns))

    for f in files:
        if "pairs" not in f:
            continue
        try: 
            import pickle
        
            device= "cuda:0"
            
            with open(f, 'rb') as fh:
                    
                coarsener = pickle.load(fh) 
            node_target_type = get_node_type(f)
            if not node_target_type:
                continue
            original_graph = coarsener.original_graph
            coarsend_graph = coarsener.summarized_graph

            #coarsend_graph = coarsend_graph.cpu()
            mapping = coarsener.get_mapping(node_target_type)
            coarsener.make_mask(mapping, node_target_type)

            labels = coarsener.get_labels(mapping, node_target_type)
            coarsend_graph.nodes[node_target_type].data["label"] = torch.tensor([labels[i] for i in range(len(labels)) ],  device=coarsend_graph.device) #,
            accur = []
            for i in range(5):
                _, coar, _, loss_coar ,_,_= run_experiments(original_graph, coarsend_graph,  model,
                                                                model_param={"hidden_dim": 64,"num_layers":4},
                                        optimizer_param={"lr": 0.01, "weight_decay": 5e-4},
                                        num_runs=1, epochs=400,eval_interval=1, target_node_type=node_target_type, run_orig=False)
            #orig_short = [ o[-1] for o in orig ]
                coar_short = [ o[-1] for o in coar ]
                accur.append(max(coar_short))
            ratio = f.split('/')[2]
            column = f.split('/')[1]
            df = update_row_by_ratio(df, columns, ratio, column,accur  )
            
            df.to_csv('res_actor.csv')            

        except Exception as e:
            print('error' , e)


eval()


#nohup python run.py > output.log 2>&1 &