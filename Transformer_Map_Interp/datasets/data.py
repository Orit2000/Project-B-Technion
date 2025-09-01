import os
import numpy as np
import torch

class SpatialDataset(torch.utils.data.Dataset):
    """A dataset class for spatial data."""

    def __init__(self, coords, features, y):
        """
        Args:
            coords: tensor with shape `(n, 2)`, coordinates of `n` instances
            features: tensor with shape `(n, d)`, `d` dimensional feature vectors of `n` instances
            y: tensor with shape `(n, )`, labels of `n` instances. Please provide zeros if unknown. 
            neighbors: tensor with shape `(n, num_neighbors)`, neighbors in an external training set. 
                       It can be none and computed later.  
        """
        super(SpatialDataset, self).__init__()

        if coords.shape[0] != features.shape[0] or features.shape[0] != y.shape[0]:
            raise Exception(f"Coordinates, features, and labels have different numbers of instances: \
                             coords.shape[0]={coords.shape[0]}, features.shape[0]={features.shape[0]}, \
                             y.shape[0]={y.shape[0]}")

        
        self.coords = torch.Tensor(coords)
        self.features = torch.Tensor(features)
        self.y = torch.Tensor(y) 
        self.y_mean = torch.mean(torch.Tensor(y))
        self.y_std = torch.std(torch.Tensor(y))
        self.obs_coords = None
        self.obs_y = None
        self.query_coords = None
        self.query_y = None
        self.obs_mask = None
        self.query_mask = None


    def __len__(self):
        return self.coords.shape[0] 

    def __getitem__(self, idx):
        
        ins = (self.coords[idx], self.features[idx], self.y[idx])

        return ins





