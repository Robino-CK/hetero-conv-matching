import torch
import random
import numpy as np


def create_random_mask(original_graph, device, target_ntype, train_ratio=0.6, val_ratio=0.2):
    # Assuming 'original_graph.nodes['movie'].data' contains the nodes
    # We will create masks for training, validation, and testing

    # 1. Get the total number of nodes (movie nodes)
    total_nodes = original_graph.nodes[target_ntype].data['feat'].shape[0]

    # 2. Shuffle the node indices
    node_indices = np.arange(total_nodes)
    np.random.shuffle(node_indices)

    # 3. Define the split proportions (e.g., 60% train, 15% val, 15% test)
    train_size = int(train_ratio * total_nodes)
    val_size = int(val_ratio * total_nodes)
    test_size = total_nodes - train_size - val_size

    # 4. Assign the splits
    train_indices = node_indices[:train_size]
    val_indices = node_indices[train_size:train_size + val_size]
    test_indices = node_indices[train_size + val_size:]

    # 5. Create masks (assuming that the original graph has the required fields for storing masks)
    train_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)

    # Set the masks
    train_mask[train_indices] = 1
    val_mask[val_indices] = 1
    test_mask[test_indices] = 1

    # Add the masks to your graph
    original_graph.nodes[target_ntype].data['train_mask'] = train_mask
    original_graph.nodes[target_ntype].data['val_mask'] = val_mask
    original_graph.nodes[target_ntype].data['test_mask'] = test_mask
    return original_graph
    # Now you have train, val, and test masks for your nodes
