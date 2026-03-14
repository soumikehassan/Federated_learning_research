"""
utils/data_partitioner.py
Handles Dirichlet distribution-based data partitioning for Federated Learning.
Allows creating IID and Non-IID (label skew) splits.
"""

import numpy as np
import torch
from torch.utils.data import Subset
import logging

logger = logging.getLogger(__name__)

class DirichletPartitioner:
    def __init__(self, dataset, num_clients=5, alpha=0.5, seed=42):
        """
        Args:
            dataset: The full dataset object (must have .samples or similar target info)
            num_clients: Number of clients to split data into
            alpha: Dirichlet parameter (alpha -> infinity means IID, alpha -> 0 means extreme Non-IID)
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Extract targets if available
        if hasattr(dataset, 'samples'):
            self.targets = np.array([s[1] for s in dataset.samples])
        elif hasattr(dataset, 'targets'):
            self.targets = np.array(dataset.targets)
        else:
            # Try to iterate (slow backup)
            self.targets = np.array([dataset[i][1] for i in range(len(dataset))])
            
        self.num_classes = len(np.unique(self.targets))

    def partition(self):
        """
        Partitions the dataset indices according to Dirichlet distribution.
        Returns: List of lists (each sublist contains indices for one client)
        """
        if self.alpha is None or self.alpha <= 0:
            # IID split (fallback if alpha is specifically None, though alpha=inf is preferred)
            indices = np.arange(len(self.dataset))
            self.rng.shuffle(indices)
            return np.array_split(indices, self.num_clients)

        min_size = 0
        min_require_size = 10
        
        indices_per_class = [np.where(self.targets == c)[0] for c in range(self.num_classes)]
        client_indices = [[] for _ in range(self.num_clients)]

        # Ensure each client gets at least some data
        while min_size < min_require_size:
            client_indices = [[] for _ in range(self.num_clients)]
            for c in range(self.num_classes):
                idx_c = indices_per_class[c]
                self.rng.shuffle(idx_c)
                
                # Sample proportions from Dirichlet
                proportions = self.rng.dirichlet([self.alpha] * self.num_clients)
                
                # Adjust proportions to ensure all samples are used
                proportions = np.array([p * (len(idx_c) < self.num_clients / p) for p in proportions])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
                
                # Split class indices among clients
                split_indices = np.split(idx_c, proportions)
                for i in range(self.num_clients):
                    client_indices[i].extend(split_indices[i].tolist())
            
            min_size = min([len(idx) for idx in client_indices])

        # Final shuffle for each client
        for i in range(self.num_clients):
            self.rng.shuffle(client_indices[i])
            
        return client_indices

def get_dirichlet_subsets(dataset, num_clients=5, alpha=0.5, seed=42):
    """
    Helper to return actual Subset objects.
    """
    partitioner = DirichletPartitioner(dataset, num_clients, alpha, seed)
    client_indices = partitioner.partition()
    return [Subset(dataset, idx) for idx in client_indices]
