import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import HeteroData
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
import torch

import torch.nn as nn


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import HeteroData
import copy
def run_experiments_unified(original_graph, coarsened_graph, model_class, num_runs=5,
                            epochs=100, optimizer=torch.optim.Adam, target_node_type="movie",
                            model_param={"hidden_dim": 256},
                            optimizer_param={"lr": 0.01, "weight_decay": 5e-4},
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            eval_interval=10, run_orig=True,
                            early_stopping_patience=30,
                            framework="dgl"):  # or "pyg"

    import copy

    def dgl_to_pyg(g):
        x_dict = {ntype: g.nodes[ntype].data['feat'] for ntype in g.ntypes}
        pyg_data = HeteroData()
        for ntype in g.ntypes:
            pyg_data[ntype].x = x_dict[ntype]
            if ntype == target_node_type:
                pyg_data[ntype].y = g.nodes[ntype].data["label"]
                pyg_data[ntype].train_mask = g.nodes[ntype].data["train_mask"]
                pyg_data[ntype].test_mask = g.nodes[ntype].data["test_mask"]
                pyg_data[ntype].val_mask = g.nodes[ntype].data["val_mask"]
        for canonical_etype in g.canonical_etypes:
            src_type, etype, dst_type = canonical_etype
            src, dst = g.edges(etype=canonical_etype)
            edge_index = torch.stack([src, dst], dim=0)
            pyg_data[(src_type, etype, dst_type)].edge_index = edge_index
        return pyg_data

    original_accuracies, coarsened_accuracies = [], []
    original_loss, coarsened_loss = [], []

    for run in range(num_runs):
        if framework == "pyg":
            original_graph = dgl_to_pyg(original_graph)
            coarsened_graph = dgl_to_pyg(coarsened_graph)
            feat_orig = original_graph.x_dict
            feat_coar = coarsened_graph.x_dict
            edge_types_orig = original_graph.edge_types
            node_types_orig = original_graph.node_types
            edge_types_coar = coarsened_graph.edge_types
            node_types_coar = coarsened_graph.node_types
        else:
            feat_orig = {ntype: original_graph.nodes[ntype].data['feat'] for ntype in original_graph.ntypes}
            feat_coar = {ntype: coarsened_graph.nodes[ntype].data['feat'] for ntype in coarsened_graph.ntypes}

        def get_indices(g):
            return (
                torch.nonzero(g[target_node_type].train_mask).squeeze(),
                torch.nonzero(g[target_node_type].val_mask).squeeze(),
                torch.nonzero(g[target_node_type].test_mask).squeeze(),
                g[target_node_type].y
            )

        if framework == "pyg":
            train_idx_orig, val_idx_orig, test_idx_orig, labels_orig = get_indices(original_graph)
            train_idx_coar, val_idx_coar, test_idx_coar, labels_coar = get_indices(coarsened_graph)
        else:
            train_idx_orig = torch.nonzero(original_graph.nodes[target_node_type].data["train_mask"]).squeeze()
            val_idx_orig = torch.nonzero(original_graph.nodes[target_node_type].data["val_mask"]).squeeze()
            test_idx_orig = torch.nonzero(original_graph.nodes[target_node_type].data["test_mask"]).squeeze()
            labels_orig = original_graph.nodes[target_node_type].data['label']

            train_idx_coar = torch.nonzero(coarsened_graph.nodes[target_node_type].data["train_mask"]).squeeze()
            val_idx_coar = torch.nonzero(coarsened_graph.nodes[target_node_type].data["val_mask"]).squeeze()
            test_idx_coar = torch.nonzero(coarsened_graph.nodes[target_node_type].data["test_mask"]).squeeze()
            labels_coar = coarsened_graph.nodes[target_node_type].data['label']

        output_dim = len(torch.unique(labels_orig))

        if framework == "pyg":
            model_original = model_class(hidden_channels=model_param.get("hidden_dim", 64),
                                         out_channels=output_dim, num_layers=4,
                                         edge_types=edge_types_orig, node_types=node_types_orig,
                                         target_node_type=target_node_type).to(device)
            model_coarsened = model_class(hidden_channels=model_param.get("hidden_dim", 64),
                                          out_channels=output_dim, num_layers=4,
                                          edge_types=edge_types_coar, node_types=node_types_coar,
                                          target_node_type=target_node_type).to(device)
        else:
            metadata_orig = (original_graph.ntypes, original_graph.etypes)
            metadata_coar = (coarsened_graph.ntypes, coarsened_graph.etypes)
            model_original = model_class(metadata=metadata_orig, x_dict=feat_orig,
                                         target_feat=target_node_type, out_dim=output_dim,
                                         **model_param).to(device)
            model_coarsened = model_class(metadata=metadata_coar, x_dict=feat_coar,
                                          target_feat=target_node_type, out_dim=output_dim,
                                          **model_param).to(device)

        best_model_originial = model_original
        best_model_coarsend = model_coarsened
        optimizer_original = optimizer(model_original.parameters(), **optimizer_param)
        optimizer_coarsened = optimizer(model_coarsened.parameters(), **optimizer_param)

        def train(model, graph, feats, labels, train_idx, optimizer):
            model.train()
            optimizer.zero_grad()
            if framework == "pyg":
                out = model(graph.x_dict, graph.edge_index_dict)  # returns tensor
            else:
                out = model(graph, feats)[target_node_type]       # returns dict
        
            if framework == "pyg":
                pred = out
                loss = F.cross_entropy(pred[train_idx], labels[train_idx])
            else:
                loss = F.cross_entropy(out[train_idx], labels[train_idx])
        
            loss.backward()
            optimizer.step()
            return loss.item()

        def test(model, graph, feats, labels, idx):
            model.eval()
            with torch.no_grad():
                if framework == "pyg":
                    out = model(graph.x_dict, graph.edge_index_dict)  # tensor
                else:
                    out = model(graph, feats)[target_node_type]       # dict
        
                if framework == "pyg":
                    pred = out.argmax(dim=1)
                else:
                    pred = out.argmax(dim=1)
        
                acc = (pred[idx] == labels[idx]).sum().item() / idx.shape[0]
                return acc

        best_val_acc_orig = 0
        best_val_acc_coar = 0
        patience_counter_orig = 0
        patience_counter_coar = 0
        stop_orig = False
        stop_coar = False

        accs_orig = []
        accs_coar = []
        losses_orig = []
        losses_coar = []

        for epoch in range(epochs):
            if run_orig and not stop_orig:
                loss_orig = train(model_original, original_graph, feat_orig, labels_orig, train_idx_orig, optimizer_original)
            elif run_orig:
                loss_orig = 0.0

            if not stop_coar:
                loss_coar = train(model_coarsened, coarsened_graph, feat_coar, labels_coar, train_idx_coar, optimizer_coarsened)
            else:
                loss_coar = 0.0

            if epoch % eval_interval == 0:
                if run_orig:
                    val_acc_orig = test(model_original, original_graph, feat_orig, labels_orig, val_idx_orig)
                    test_acc_orig = test(best_model_originial, original_graph, feat_orig, labels_orig, test_idx_orig)
                    accs_orig.append(test_acc_orig)
                    losses_orig.append(loss_orig)
                    if not stop_orig:
                        if val_acc_orig > best_val_acc_orig:
                            best_val_acc_orig = val_acc_orig
                            patience_counter_orig = 0
                            best_model_originial = copy.deepcopy(model_original)
                        else:
                            patience_counter_orig += 1
                            if patience_counter_orig >= early_stopping_patience:
                                stop_orig = True

                val_acc_coar = test(model_coarsened, coarsened_graph, feat_coar, labels_coar, val_idx_coar)
                test_acc_coar = test(best_model_coarsend, coarsened_graph, feat_coar, labels_coar, test_idx_coar)
                accs_coar.append(test_acc_coar)
                losses_coar.append(loss_coar)
                if not stop_coar:
                    if val_acc_coar > best_val_acc_coar:
                        best_val_acc_coar = val_acc_coar
                        patience_counter_coar = 0
                        best_model_coarsend = copy.deepcopy(model_coarsened)
                    else:
                        patience_counter_coar += 1
                        if patience_counter_coar >= early_stopping_patience:
                            stop_coar = True

        if run_orig:
            final_acc_orig = test(best_model_originial, original_graph, feat_orig, labels_orig, test_idx_orig)
            accs_orig.append(final_acc_orig)
            original_accuracies.append(accs_orig)
            original_loss.append(losses_orig)

        final_acc_coar = test(best_model_coarsend, coarsened_graph, feat_coar, labels_coar, test_idx_coar)
        accs_coar.append(final_acc_coar)
        coarsened_accuracies.append(accs_coar)
        coarsened_loss.append(losses_coar)

    return original_accuracies, coarsened_accuracies, original_loss, coarsened_loss