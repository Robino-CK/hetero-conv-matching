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
from Models.Han import HAN

from Experiments.model_helper import run_experiments_unified
from Projections.AutoEncoder import MultiviewAutoencoder
import os
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os

import pickle

def make_mask(self, mapping, ntype, is_acm =False):
    labels_dict = dict()
    inverse_mapping = dict()
    if is_acm:
        # Original test mask
        test_mask = self.original_graph.nodes[ntype].data["test_mask"]
        
        # Get indices where test_mask is True
        test_indices = torch.nonzero(test_mask, as_tuple=True)[0]
        
        # Shuffle and select 20% of them
        num_val = int(0.2 * len(test_indices))
        perm = torch.randperm(len(test_indices))
        val_indices = test_indices[perm[:num_val]]
        
        # Initialize val_mask with False and set selected indices to True
        val_mask = torch.zeros_like(test_mask, dtype=torch.bool)
        val_mask[val_indices] = True
        
        # Set the selected indices to False in test_mask
        test_mask[val_indices] = False
        
        # Save the updated masks back
        self.original_graph.nodes[ntype].data["test_mask"] = test_mask
        self.original_graph.nodes[ntype].data["val_mask"] = val_mask
        self.summarized_graph.nodes[ntype].data["val_mask"] = torch.zeros_like(self.summarized_graph.nodes[ntype].data["test_mask"], dtype=torch.bool)
        self.summarized_graph.nodes[ntype].data["test_mask"] = torch.zeros_like(self.summarized_graph.nodes[ntype].data["test_mask"], dtype=torch.bool)
    for ori_node, coar_node in mapping.items():
        if coar_node in inverse_mapping:
            inverse_mapping[coar_node].append(ori_node)
        else:
            inverse_mapping[coar_node] = [ori_node]
            
    for coar_node, ori_list in inverse_mapping.items():
        label_list = []
        for ori_node in ori_list:
            label_list.append(self.original_graph.nodes[ntype].data["train_mask"][ori_node].item())
        is_train = torch.any(torch.tensor(label_list, device=self.device))    
        
        is_val = False
        for ori_node in ori_list:
            
            is_val = (self.original_graph.nodes[ntype].data["val_mask"][ori_node] or is_val ) and not is_train
         
    # print(coar_node)
        self.summarized_graph.nodes[ntype].data["train_mask"][coar_node] = is_train
        is_test =  (not is_val and not is_train)
        # print(is_val, is_train, is_test)
        self.summarized_graph.nodes[ntype].data["test_mask"][coar_node] =is_test

        
        
        self.summarized_graph.nodes[ntype].data["val_mask"][coar_node] = is_val
from collections import Counter
def get_labels(self, mapping, ntype):
    #self.make_mask(mapping, ntype)
    labels_dict = dict()
    inverse_mapping = dict()
    for ori_node, coar_node in mapping.items():
        if coar_node in inverse_mapping:
            inverse_mapping[coar_node].append(ori_node)
        else:
            inverse_mapping[coar_node] = [ori_node]
    for coar_node, ori_list in inverse_mapping.items():
        label_list = []
        # if not self.summarized_graph.nodes[ntype].data["train_mask"][coar_node]:
        #     label_list.append(-1)
        # else:
        for ori_node in ori_list:
            if self.original_graph.nodes[ntype].data["train_mask"][ori_node] or not self.summarized_graph.nodes[ntype].data["train_mask"][coar_node]:
                label_list.append(self.original_graph.nodes[ntype].data["label"][ori_node].item())

        counter = Counter(label_list)
        
        labels_dict[coar_node],_ = counter.most_common()[0]
    
    return labels_dict

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
    elif "cora" in name.lower():
        return "paper"
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
def get_model(name):
    if name == 'SGCN':
        return HeteroSGCPaper
    if name == 'HAN':
        return HAN
    if name == "GCN":
        return ImprovedGCN
def eval( model_name='SGCN' , device='cuda:0', lr=0.001): 
    files = get_all_files('results') #
    #print(files)
    model = get_model(model_name)
    columns = set()
    columns.add('ratio')
    for f in files:
        columns.add(f.split('/')[1])
    
    df = pd.DataFrame(columns=list(columns))

    for f in files:
        #if "pair" in f.lower()  or not "cora" in f.lower():
        # if not "acm" in f.lower() and not "dblp" in f.lower() and not "imdb" in f.lower() :
        #     print("no", f)
        #     continue
        # if not "TRI" in f and not 'zscore' in f:
        #     continue
        target = ["0.1", "0.3", "0.5", "1.0"]
        if not any(x in f for x in target):
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
                    

                    #coarsend_graph = coarsend_graph.cpu()
                    mapping = coarsener.get_mapping(node_target_type)
                    make_mask(coarsener, mapping, node_target_type, "ACM" in f)
                    
                    labels = get_labels(coarsener, mapping, node_target_type)
                    original_graph = coarsener.original_graph.to(device)
                    coarsend_graph = coarsener.summarized_graph.to(device)

                    coarsend_graph.nodes[node_target_type].data["label"] = torch.tensor([labels[i] for i in range(len(labels)) ],  device=coarsend_graph.device) #,
                    # print("ratio", coarsend_graph.num_nodes()/ original_graph.num_nodes() ) 

                    
                   # mapping = coarsener.get_mapping(node_target_type)
                    #coarsener.make_mask(mapping, node_target_type)

                    #labels = coarsener.get_labels(mapping, node_target_type)
                    #coarsend_graph.nodes[node_target_type].data["label"] = torch.tensor([labels[i] for i in range(len(labels)) ],  device=coarsend_graph.device) #,
                    
                accur = []
                for i in range(10):
                    print(i)
                    if model_name == "HAN":
                        _,acc,_,_ = run_experiments_unified(original_graph, coarsend_graph,  model,
                                                                        model_param={"hidden_dim": 64,"num_layers":4},
                                                optimizer_param={"lr": lr, "weight_decay": 5e-4}, device=device,
                                                num_runs=1, epochs=100,eval_interval=1, target_node_type=node_target_type, run_orig=False, framework="pyg")
                    else:
                        _,acc,_,_ = run_experiments_unified(original_graph, coarsend_graph,  model,
                                                                        model_param={"hidden_dim": 64,"num_layers":4},
                                                optimizer_param={"lr": lr, "weight_decay": 5e-4}, device=device,
                                                num_runs=1, epochs=100,eval_interval=1, target_node_type=node_target_type, run_orig=False)
                #orig_short = [ o[-1] for o in orig ]

                    accur.append(acc[0][-1])  
                    #print(max(acc[0]))
                ratio = f.split('/')[2]
                column = f.split('/')[1]
                
                df = update_row_by_ratio(df, columns, ratio, column,accur  )
                df.to_csv(f'master_table_{model_name}.csv')      

                del original_graph, coarsend_graph, labels, mapping
            
         #   torch.cuda.empty_cache()
            
                  

        except Exception as e:
            print('error' , e)


#eval("GCN", lr=0.01)
#eval("GCN", lr=0.005)
eval("GCN", lr=0.001)
#eval("HAN")
eval("SGCN")


#nohup python run.py > output.log 2>&1 &
