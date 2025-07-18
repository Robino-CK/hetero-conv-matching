import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import HeteroData
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import HeteroData

def run_experiments(original_graph, coarsend_graph, model_class, num_runs=5,
                    epochs=100, optimizer=torch.optim.Adam, target_node_type="movie",
                    model_param={"hidden_dim": 256},
                    optimizer_param={"lr": 0.01, "weight_decay": 5e-4},
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    eval_interval=10, run_orig=True,
                    early_stopping_patience=10):

    original_accuracies = []
    coarsened_accuracies = []
    original_loss = []
    coarsened_loss = []

    for run in range(num_runs):
        train_idx_orig = torch.nonzero(original_graph.nodes[target_node_type].data["train_mask"]).squeeze()
        test_idx_orig = torch.nonzero(original_graph.nodes[target_node_type].data["test_mask"]).squeeze()
        val_idx_orig = torch.nonzero(original_graph.nodes[target_node_type].data["test_mask"]).squeeze()
        labels_orig = original_graph.nodes[target_node_type].data['label']

        train_idx_coar = torch.nonzero(coarsend_graph.nodes[target_node_type].data["train_mask"]).squeeze()
        test_idx_coar = torch.nonzero(coarsend_graph.nodes[target_node_type].data["test_mask"]).squeeze()
        val_idx_coar = torch.nonzero(coarsend_graph.nodes[target_node_type].data["test_mask"]).squeeze()
        labels_coar = coarsend_graph.nodes[target_node_type].data['label']

        metadata_orig = (original_graph.ntypes, original_graph.etypes)
        metadata_coar = (coarsend_graph.ntypes, coarsend_graph.etypes)

        feat_orig = {ntype: original_graph.nodes[ntype].data['feat'] for ntype in original_graph.ntypes}
        feat_coar = {ntype: coarsend_graph.nodes[ntype].data['feat'] for ntype in coarsend_graph.ntypes}
        
        output_dim = len(torch.unique(labels_orig))
        model_original = model_class(metadata=metadata_orig,
                                     x_dict=feat_orig,
                                     target_feat=target_node_type,
                                     out_dim=output_dim,
                                     **model_param).to(device)
        optimizer_original = optimizer(model_original.parameters(), **optimizer_param)

        model_coarsened = model_class(metadata=metadata_coar,
                                      x_dict=feat_coar,
                                      target_feat=target_node_type,
                                      out_dim=output_dim,
                                      **model_param).to(device)
        optimizer_coarsened = optimizer(model_coarsened.parameters(), **optimizer_param)

        def train(model, graph, feats, labels, train_idx, optimizer):
            model.train()
            optimizer.zero_grad()
            logits = model(graph, feats)[target_node_type]
            loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()
            return loss

        def test(model, graph, feats, labels, val_idx):
            model.eval()
            with torch.no_grad():
                logits = model(graph, feats)[target_node_type]
                preds = logits.argmax(dim=1)
                correct = (preds[val_idx] == labels[val_idx]).sum().item()
                return correct / val_idx.shape[0], preds

        original_acc_per_run = []
        coarsened_acc_per_run = []
        original_loss_per_run = []
        coarsened_loss_per_run = []

        best_val_acc_orig = 0
        best_val_acc_coar = 0
        patience_counter_orig = 0
        patience_counter_coar = 0
        stop_orig = False
        stop_coar = False

        for epoch in range(epochs):
            # Training only if not early-stopped
            if run_orig and not stop_orig:
                loss_orig = train(model_original, original_graph, feat_orig, labels_orig, train_idx_orig, optimizer_original)
            elif run_orig:
                loss_orig = torch.tensor(0.0)  # Dummy loss for shape consistency

            if not stop_coar:
                loss_coar = train(model_coarsened, coarsend_graph, feat_coar, labels_coar, train_idx_coar, optimizer_coarsened)
            else:
                loss_coar = torch.tensor(0.0)  # Dummy loss

            # Evaluate every eval_interval
            if epoch % eval_interval == 0:
                if run_orig:
                    val_acc_orig, _ = test(model_original, original_graph, feat_orig, labels_orig, val_idx_orig)
                    original_acc_per_run.append(val_acc_orig)
                    original_loss_per_run.append(loss_orig.item())

                    if not stop_orig:
                        if val_acc_orig > best_val_acc_orig:
                            best_val_acc_orig = val_acc_orig
                            patience_counter_orig = 0
                        else:
                            patience_counter_orig += 1
                            if patience_counter_orig >= early_stopping_patience:
                                stop_orig = True

                val_acc_coar, _ = test(model_coarsened, original_graph, feat_orig, labels_orig, val_idx_orig)
                coarsened_acc_per_run.append(val_acc_coar)
                coarsened_loss_per_run.append(loss_coar.item())

                if not stop_coar:
                    if val_acc_coar > best_val_acc_coar:
                        best_val_acc_coar = val_acc_coar
                        patience_counter_coar = 0
                    else:
                        patience_counter_coar += 1
                        if patience_counter_coar >= early_stopping_patience:
                            stop_coar = True

        # Final evaluation
        if run_orig:
            final_acc_orig, _ = test(model_original, original_graph, feat_orig, labels_orig, val_idx_orig)
            original_acc_per_run.append(final_acc_orig)
            original_accuracies.append(original_acc_per_run)
            original_loss.append(original_loss_per_run)

        final_acc_coar, _ = test(model_coarsened, original_graph, feat_orig, labels_orig, val_idx_orig)
        coarsened_acc_per_run.append(final_acc_coar)
        coarsened_accuracies.append(coarsened_acc_per_run)
        coarsened_loss.append(coarsened_loss_per_run)

    return original_accuracies



def dgl_to_pyg(g, target_node_type="author"):
    x_dict = {ntype: g.nodes[ntype].data['feat'] for ntype in g.ntypes}
      
    pyg_data = HeteroData()

    # Add node features
    for ntype in g.ntypes:
        #print(x_dict)
        pyg_data[ntype].x = x_dict[ntype]
        if ntype == target_node_type:
            #print(g.nodes[ntype].data.keys())
            pyg_data[ntype].y = g.nodes[ntype].data["label"]
            pyg_data[ntype].train_mask = g.nodes[ntype].data["train_mask"]
            pyg_data[ntype].test_mask = g.nodes[ntype].data["test_mask"]
            pyg_data[ntype].val_mask = g.nodes[ntype].data["test_mask"]
        

    # Add edge index for each edge type
    for canonical_etype in g.canonical_etypes:
        src_type, etype, dst_type = canonical_etype
        src, dst = g.edges(etype=canonical_etype)
        edge_index = torch.stack([src, dst], dim=0)
        pyg_data[(src_type, etype, dst_type)].edge_index = edge_index

    return pyg_data


def run_experiments_pyg(original_graph, coarsend_graph, model_class, num_runs=5,
                    epochs=1, optimizer=torch.optim.Adam, target_node_type="movie",
                    model_param={"hidden_dim": 256},
                    optimizer_param={"lr": 0.01, "weight_decay": 5e-4},
                    device="cuda" if torch.cuda.is_available() else "cpu", eval_interval=10, run_orig=True):
    


    for run in range(num_runs):
        train_idx_orig = torch.nonzero(original_graph.nodes[target_node_type].data["train_mask"]).squeeze()
        test_idx_orig = torch.nonzero(original_graph.nodes[target_node_type].data["test_mask"]).squeeze()
        val_idx_orig = torch.nonzero(original_graph.nodes[target_node_type].data["test_mask"]).squeeze()
        labels_orig = original_graph.nodes[target_node_type].data['label']

        train_idx_coar = torch.nonzero(coarsend_graph.nodes[target_node_type].data["train_mask"]).squeeze()
        test_idx_coar = torch.nonzero(coarsend_graph.nodes[target_node_type].data["test_mask"]).squeeze()
        val_idx_coar = torch.nonzero(coarsend_graph.nodes[target_node_type].data["test_mask"]).squeeze()
        labels_coar = coarsend_graph.nodes[target_node_type].data['label']

        pygorig = dgl_to_pyg(original_graph, target_node_type=target_node_type)
        pygcoar = dgl_to_pyg(coarsend_graph, target_node_type)
        node_types, edge_types = pygorig.metadata()
      
        
        output_dim = len(torch.unique(labels_orig))
        model_original = model_class( hidden_channels=64, out_channels=output_dim, num_layers=4, 
                 edge_types=edge_types, node_types=node_types, target_node_type= target_node_type).to(device)
        optimizer_original = optimizer(model_original.parameters(), **optimizer_param)
        node_types, edge_types = pygcoar.metadata()
        model_coarsened = model_class(hidden_channels=64, out_channels=output_dim, num_layers=4, 
                 edge_types=edge_types, node_types=node_types, target_node_type= target_node_type).to(device)
        optimizer_coarsened = optimizer(model_coarsened.parameters(), **optimizer_param)
        criterion = nn.CrossEntropyLoss()
        def train():
            model_coarsened.train()
            optimizer_coarsened.zero_grad()
            out = model_coarsened(pygcoar.x_dict, pygcoar.edge_index_dict)
            loss = criterion(out[pygcoar[target_node_type].train_mask],
                            pygcoar[target_node_type].y[pygcoar[target_node_type].train_mask])
            loss.backward()
            optimizer_coarsened.step()
            return loss.item()
        def test_on(graph):
            model_coarsened.eval()
            with torch.no_grad():
                out = model_coarsened(graph.x_dict, graph.edge_index_dict)
                pred = out.argmax(dim=1)

                accs = []
                for split in ['train_mask', 'val_mask', 'test_mask']:
                    mask = graph[target_node_type][split]
                    acc = (pred[mask] == graph[target_node_type].y[mask]).sum().item() / mask.sum().item()
                    accs.append(acc)
                return accs  # train_acc, val_acc, test_acc
            
        best_val = 0
        for epoch in range(1, epochs + 1):
            loss = train()
            train_acc, val_acc, test_acc = test_on(pygorig)
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train: {train_acc:.4f} | "
                    f"Val: {val_acc:.4f} | Test: {test_acc:.4f}")
        
                if val_acc > best_val:
                    best_val = val_acc
      
        return [best_val]