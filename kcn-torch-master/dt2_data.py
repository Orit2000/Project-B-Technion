import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
from data import SpatialDataset
import os

class DT2Dataset(Dataset):
    """Dataset for DTED elevation maps in SpatialDataset format."""
    
    def __init__(self, dt2_file, include_coords_in_features=True, normalize=True):
        """
        Args:
            dt2_file: path to the .dt2 file
            include_coords_in_features: if True, adds lat/lon as part of the feature vector
            normalize: if True, normalize the feature values
        """
        with rasterio.open(dt2_file) as src:
            elevation = src.read(1)  # shape: (height, width)
            transform = src.transform
            height, width = elevation.shape

            # Create coordinate grid
            lon_coords = np.array([transform[2] + i * transform[0] for i in range(width)])
            lat_coords = np.array([transform[5] + j * transform[4] for j in range(height)])
            lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

            # Flatten all arrays
            coords = np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=1)  # [n, 2]
            elevations = elevation.flatten().astype(np.float32).reshape(-1, 1)    # [n, 1]

            # Features: only elevation for now
            if include_coords_in_features:
                features = np.concatenate([coords, elevations], axis=1)
            else:
                features = elevations

            # Labels: can be elevation again, or zeros if unknown
            y = elevations  # or y = np.zeros_like(elevations) if elevation is not target

            # Convert to tensors
            self.coords = torch.from_numpy(coords).float()
            self.features = torch.from_numpy(features).float()
            self.y = torch.from_numpy(y).float()

            if normalize:
                self._normalize_features()

    def _normalize_features(self):
        self.feature_mean = self.features.mean(dim=0, keepdim=True)
        self.feature_std = self.features.std(dim=0, keepdim=True) + 1e-6
        self.features = (self.features - self.feature_mean) / self.feature_std

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx], self.features[idx], self.y[idx]

def load_dt2_data(args):
    """
    Load data for training and testing from DT2 (elevation) files

    Args
    ----
    args : will use three fields, args.dataset, args.data_path, args.random_seed  

    Returns
    -------
    coords    : np.ndarray, shape (N, 2), coordinates of the data points
    features  : np.ndarray, shape (N, D), features of the data points
    y         : np.ndarray, shape (N, 1), labels of the data points
    num_total_train : int, number of training data points. The first `num_total_train` 
                      of instances from three other return values should form the training set
    """

    # data file path
    #dt2_file = os.path.join(args.data_path, args.dataset + ".dt2")
    # FOR COLLAB:
    dt2_file= '/content/Project-B-Technion/kcn-torch-master/datasets/n32_e035_1arc_v3.dt2'
    # check if file exists
    if not os.path.isfile(dt2_file):
        #raise Exception(f"DT2 file {dt2_file} not found. Please provide the correct file.")
        raise Exception(f"{dt2_file}")

    # Create DT2Dataset object
    dataset = DT2Dataset(dt2_file=dt2_file, include_coords_in_features=True, normalize=True)
    # Split into train and test sets
    num_total_train = int(dataset.coords.shape[0] * 0.8)  # Use 80% for training

    # Random shuffle and split
    perm = np.random.RandomState(seed=args.random_seed).permutation(dataset.coords.shape[0])
    trainset = SpatialDataset(
        coords=dataset.coords[perm[:num_total_train]].numpy(),
        features=dataset.features[perm[:num_total_train]].numpy(),
        y=dataset.y[perm[:num_total_train]].numpy()
    )
    testset = SpatialDataset(
        coords=dataset.coords[perm[num_total_train:]].numpy(),
        features=dataset.features[perm[num_total_train:]].numpy(),
        y=dataset.y[perm[num_total_train:]].numpy()
    )

    # Feature normalization is already handled in DT2Dataset, so no need to repeat
    return trainset, testset
