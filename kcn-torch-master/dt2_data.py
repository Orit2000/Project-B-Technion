import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
from data import SpatialDataset
import os
from matplotlib import pyplot as plt

class DT2Dataset(Dataset):
    """Dataset for DTED elevation maps in SpatialDataset format."""
    
    def __init__(self, dt2_file, include_elevation_in_features=False, normalize=True):
        """
        Args:
            dt2_file: path to the .dt2 file
            include_coords_in_features: if True, adds lat/lon as part of the feature vector
            normalize: if True, normalize the feature values
            
            Notes:
            transform is a 2D affine transformation that maps pixel coordinates (row, col) to geographic coordinates (lon, lat).
            transform = (pixel_width, row_rotation, x_min, col_rotation, pixel_height, y_max).
            transform[0] is pixel width in degrees (Δlon)
            transform[2] is lon_min (start longitude)
            transform[4] is pixel height (negative, because images start from top-left)
            transform[5] is lat_max (top latitude)

        """
        with rasterio.open(dt2_file) as src:
            print("I am reading!")
            elevation = src.read(1)  # shape: (height, width). elevation[i,j] - gives the elevation in meters at (i,j)
            transform = src.transform
            height, width = elevation.shape

            # Create coordinate grid
            lon_coords = np.array([transform[2] + i * transform[0] for i in range(width)]) # 1D array of longitudes for each column
            lat_coords = np.array([transform[5] + j * transform[4] for j in range(height)]) # 1D array of latitudes for each row
            lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords) # Creates full grids of shape (height, width) — so now each pixel has an exact lat-lon pair.

            # Flatten all arrays
            coords = np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=1)  # [n, 2]
            elevations = elevation.flatten().astype(np.float32).reshape(-1, 1)    # [n, 1]

            # Features: only coords for now
            if include_elevation_in_features:
                features = np.concatenate([coords, elevations], axis=1)
                print(f"Features shape:{features.shape}")
            else:
                features = coords

            # Labels
            y = elevations  
                
            # Convert to tensors
            self.coords = torch.from_numpy(coords).float()
            self.features = torch.from_numpy(features).float()
            self.y = torch.from_numpy(y).float()
            
            #if normalize: #CHECK: Should I normalize the coords?
            #    self._normalize_features()

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
    cache_path = f"cache/trainset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt"
    dt2_file = os.path.join(args.data_path, args.dataset + ".dt2")
    print(f"[DEBUG] Using dt2_file path: {dt2_file}")
    assert os.path.isfile(dt2_file), f"File does not exist: {dt2_file}"
    if os.path.exists(cache_path):
        print("Loading cached sets...")
        trainset = torch.load(f"cache/trainset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt",weights_only=False)
        testset = torch.load(f"cache/testset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt",weights_only=False)
    else:
        print("Creating and caching sets...")
        # FOR COLLAB:
        #dt2_file= "/content/Project-B-Technion/kcn-torch-master/datasets/n32_e035_1arc_v3.dt2"
        #print(f"dt2_file is: {dt2_file}")
        # check if file exists
        #if not os.path.isfile(dt2_file):
        #raise Exception(f"DT2 file {dt2_file} not found. Please provide the correct file.")
        #raise Exception(f"{dt2_file}")
        # Create DT2Dataset object
        dataset = DT2Dataset(dt2_file=dt2_file, include_elevation_in_features=True, normalize=args.normalize_elev)
        print("dataset exists!")
        # Resample:
        total = dataset.coords.shape[0]
        #print(f"min of coord: ({min(dataset.coords[:,0])},{min(dataset.coords[:,1])})")
        #print(f"max of coord: ({max(dataset.coords[:,0])},{max(dataset.coords[:,1])})")
        keep_n = int(total * args.keep_n)
        #keep_n = int(total * 0.2)
        selected_idx = np.random.RandomState(seed=args.random_seed).choice(total, size=keep_n, replace=False)

        dataset.coords = dataset.coords[selected_idx]
        dataset.features = dataset.features[selected_idx]
        dataset.y = dataset.y[selected_idx]
        # Split into train and test sets
        num_total_train = int(dataset.coords.shape[0] * 0.8)  # Use 80% for training
        # Random shuffle and split
        perm = np.random.RandomState(seed=args.random_seed).permutation(dataset.coords.shape[0])
        trainset = SpatialDataset(
            coords=dataset.coords[perm[:num_total_train]].numpy(),
            features=dataset.features[perm[:num_total_train]].numpy(),
            y=dataset.y[perm[:num_total_train]].numpy()
        )
        
        if args.normalize_elev:
            num_total_train = len(trainset.y)
            num_valid = args.validation_size * num_total_train
            num_train = int(num_total_train - args.validation_size)
            y_mean = trainset.y[0:num_train].mean(dim=0, keepdim=True)
            y_std = trainset.y[0:num_train].std(dim=0, keepdim=True) + 1e-6
            trainset.y = (trainset.y - y_mean) / y_std
            print("NEW NORM IS COMING!")
            trainset.y_mean = y_mean # CHECK THIS WRITING
            trainset.y_std = y_std
            
        testset = SpatialDataset(
            coords=dataset.coords[perm[num_total_train:]].numpy(),
            features=dataset.features[perm[num_total_train:]].numpy(),
            y=dataset.y[perm[num_total_train:]].numpy()
        )
        testset.y = (testset.y - y_mean) / y_std 
        
        inspect_dataset(trainset, name="Train")
        inspect_dataset(testset, name="Test")
        torch.save(trainset, f"cache/trainset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt")
        torch.save(testset, f"cache/testset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt")
        #set_plot_2(trainset)
        #set_plot_2(testset)
        # Feature normalization is already handled in DT2Dataset, so no need to repeat
    return trainset, testset

def inspect_dataset(dataset, name="Train"):
    print(f"\n {name} Dataset Summary")
    print(f"➤ Number of points: {len(dataset)}")
    print(f"➤ Coords shape: {dataset.coords.shape}")
    print(f"➤ Feature shape: {dataset.features.shape}")
    print(f"➤ Label shape: {dataset.y.shape}")
    print(f"➤ Feature mean/std (first 5 dims):")
    print(f"   mu = {dataset.features.mean(0)[:5].numpy()}")
    print(f"  std = {dataset.features.std(0)[:5].numpy()}")
    print(f"➤ Elevation min/max: {dataset.y.min().item():.2f} / {dataset.y.max().item():.2f}")

    # Coordinates info
    coords = dataset.coords.numpy()
    print(f"➤ Lat range: {coords[:, 0].min():.4f} - {coords[:, 0].max():.4f}")
    print(f"➤ Lon range: {coords[:, 1].min():.4f} - {coords[:, 1].max():.4f}")

def set_plot(dataset):
    extent = [dataset.coords[1].min(),dataset.coords[1].max(), dataset.coords[0].min(), dataset.coords[0].max()]  # Get geographical extent
    plt.figure(figsize=(10, 8))
    plt.imshow(dataset.y, cmap="terrain", extent=extent, origin="upper")
    plt.colorbar(label="Elevation (m)")
    plt.title("DTED Level 2 Elevation Data")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    
def set_plot_2(dataset):
    plt.figure(figsize=(10, 8))
    extent = [dataset.coords[:, 1].min(),dataset.coords[:, 1].max(), dataset.coords[:, 0].min(), dataset.coords[:,0].max()]  # Get geographical extent
    plt.imshow(dataset.y, cmap="terrain", extent=extent, origin="upper")
    plt.colorbar(label="Elevation (m)")
    
    # Scatter your training points on top
    lats = dataset.coords[:, 0]
    lons = dataset.coords[:, 1]
    plt.scatter(lons, lats, s=2, c='red', label='Training points', alpha=0.6)

    plt.title("DTED Level 2 with Sampled Training Points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show()