import torch
import torch.nn.functional as F
def run_experiments(original_graph, coarsend_graph, model_class, num_runs=5,
                    epochs=1, optimizer=torch.optim.Adam, target_node_type="movie",
                    model_param={"hidden_dim": 256},
                    optimizer_param={"lr": 0.01, "weight_decay": 5e-4},
                    device="cuda" if torch.cuda.is_available() else "cpu", eval_interval=10):
    
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
                                     out_dim= output_dim,
                                     **model_param).to(device)
        optimizer_original = optimizer(model_original.parameters(), **optimizer_param)

        model_coarsened = model_class(metadata=metadata_coar,
                                      x_dict=feat_coar,
                                      target_feat=target_node_type,
                                      out_dim= output_dim,   
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
        
        for epoch in range(epochs):
            loss_orig = train(model_original, original_graph, feat_orig, labels_orig, train_idx_orig, optimizer_original)
            loss_coar = train(model_coarsened, coarsend_graph, feat_coar, labels_coar, train_idx_coar, optimizer_coarsened)
            if epoch % eval_interval == 0:
                original_acc, _ = test(model_original, original_graph, feat_orig, labels_orig, val_idx_orig)
                coarsened_acc, _ = test(model_coarsened, original_graph, feat_orig, labels_orig, val_idx_orig)
                original_acc_per_run.append(original_acc)
                coarsened_acc_per_run.append(coarsened_acc)

                original_loss_per_run.append(loss_orig.item())
                coarsened_loss_per_run.append(loss_coar.item())
        
                
                
        original_acc, _ = test(model_original, original_graph, feat_orig, labels_orig, val_idx_orig)
        coarsened_acc, _ = test(model_coarsened, original_graph, feat_orig, labels_orig, val_idx_orig)
        original_acc_per_run.append(original_acc)
        coarsened_acc_per_run.append(coarsened_acc)
        
        original_accuracies.append(original_acc_per_run)
        coarsened_accuracies.append(coarsened_acc_per_run)

        original_loss.append(original_loss_per_run)
        coarsened_loss.append(coarsened_loss_per_run)

    return original_accuracies, coarsened_accuracies, original_loss, coarsened_loss, model_coarsened, model_original