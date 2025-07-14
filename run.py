import dgl
import torch
import numpy as np
from Coarsener.HeteroRGCNCoarsener import HeteroRGCNCoarsener
from Data.Citeseer import Citeseer
from Data.DBLP import DBLP
from Data.IMDB import IMDB
from Data.ACM import ACM
#from Data.Actor import Actor
from Projections.ContrastiveLearner import NonLinearContrastiveLearner, LinearContrastiveLearner
from Projections.CCA import CCA
from Projections.DeepCCA import DeepCCA

from Models.SimpleHeteroGCN import HeteroGCNCiteer    
from Models.ImprovedGCN import ImprovedGCN
from Models.HeteroSage import HeteroSAGE
from Models.HeteroSGC import HeteroSGCPaper
from Models.Han import HAN

from Experiments.model_helper import run_experiments, run_experiments_pyg
from Projections.AutoEncoder import MultiviewAutoencoder
import os
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os
def torch_to_dgl(data):
    from torch_geometric.data import HeteroData


    # ... your existing graph setup
    edge_index_dict = {}
    for edge_type in data.edge_types:
        src_type, relation_type, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        edge_index_dict[(src_type, relation_type, dst_type)] = edge_index
    import dgl
    import torch

    # Compute number of nodes per node type
    num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}

    # Convert edge_index_dict to DGL format
    dgl_graph = dgl.heterograph({
        (src, src +rel+dst , dst): (edge_index[0], edge_index[1])
        for (src, rel, dst), edge_index in edge_index_dict.items()
    }, num_nodes_dict=num_nodes_dict)
    for ntype in data.node_types:
        for key, value in data[ntype].items():
            if key == 'x':
                dgl_graph.nodes[ntype].data['feat'] = value
            else:
                dgl_graph.nodes[ntype].data[key] = value

    for ntype in data.node_types:
        for key, value in data[ntype].items():
            # This includes x, y, train_mask, val_mask, test_mask, etc.
            dgl_graph.nodes[ntype].data[key] = value
    dgl_graph.nodes['author'].data['label'] = dgl_graph.nodes['author'].data['y']
    dgl_graph.nodes['author'].data['test_mask'] = ~dgl_graph.nodes['author'].data['train_mask']
    return dgl_graph
#dgl_graph = dgl_graph.to(device)

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
    elif 'imdb' in name.lower():
        return "movie"
    if 'acm' in name.lower():
        return "paper"
    elif 'dblp' in name.lower():
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
        df.loc[df['ratio'] == ratio, column_name] = [value]  # Ensure list is assigned
    else:
        # Create a new row with NaNs and set ratio and column value
        new_row = {col: None for col in columns}
        new_row['ratio'] = ratio
        new_row[column_name] = value  # Can be a list
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def eval( model=ImprovedGCN , device='cuda:0'): 
    files = get_all_files('results') #
    #print(files)
    
    columns = set()
    columns.add('ratio')
    for f in files:
        columns.add(f.split('/')[1])
    
    df = pd.DataFrame(columns=list(columns))

    for f in files:
        if not "pair" in f.lower()  or not "dblp" in f.lower():
            print("no", f)
            continue
        try: 
          # torch.cuda.empty_cache()
            import pickle
            print(f)
            node_target_type = get_node_type(f)
            if not node_target_type:
                continue
            
            with open(f, 'rb') as fh:
                if "ugc" in f:
                    dataset = DBLP() 
                    print("asd")
                    original_graph = dataset.load_graph(n_components=30)
                    device= "cuda:0"
                    original_graph = original_graph.to(device)
                    
                    #original_graph = Data.utils.create_random_mask(original_graph, device=device, target_ntype="paper") 
                    data = torch.load(fh)
                    #print(data)
                    #print(data["author"]["y"])
                    coarsend_graph = torch_to_dgl(data)
                    coarsend_graph = coarsend_graph.to(device)
                else:
                    coarsener = pickle.load(fh) 
                    
                    original_graph = coarsener.original_graph.to(device)
                    coarsend_graph = coarsener.summarized_graph.to(device)

                    #coarsend_graph = coarsend_graph.cpu()
                    mapping = coarsener.get_mapping(node_target_type)
                    coarsener.make_mask(mapping, node_target_type)

                    labels = coarsener.get_labels(mapping, node_target_type)
                    coarsend_graph.nodes[node_target_type].data["label"] = torch.tensor([labels[i] for i in range(len(labels)) ],  device=coarsend_graph.device) #,
                    
                accur = []
                for i in range(5):
                    print(i)
                    
                    acc= run_experiments(original_graph, coarsend_graph,  model,
                                                                    model_param={"hidden_dim": 64,"num_layers":4},
                                            optimizer_param={"lr": 0.01, "weight_decay": 5e-4}, device=device,
                                            num_runs=1, epochs=400,eval_interval=1, target_node_type=node_target_type, run_orig=False)
                #orig_short = [ o[-1] for o in orig ]
                    accur.append(max(acc))  
                ratio = f.split('/')[2]
                column = f.split('/')[1]
                
                df = update_row_by_ratio(df, columns, ratio, column,accur  )
                df.to_csv('run_gcn_dblp.csv')      
                del original_graph, coarsend_graph, labels, mapping
            
         #   torch.cuda.empty_cache()
            
                  

        except Exception as e:
            print('error' , e)


eval()


#nohup python run.py > output.log 2>&1 &
