import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
from data import SpatialDataset
import os
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
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

            # Features: only coords by default or coords and elevation
            '''
            if include_elevation_in_features:
                features = np.concatenate([coords, elevations], axis=1)
                print(f"Features shape:{features.shape}")
            else:
                features = coords
            '''
            features = coords
            # Maybe features = []
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
    os.makedirs("cache/", exist_ok=True)
    cache_path = f"cache/trainset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt"
    dt2_file = os.path.join(args.data_path, args.dataset + ".tiff")
    print(f"[DEBUG] Using dt2_file path: {dt2_file}")
    assert os.path.isfile(dt2_file), f"File does not exist: {dt2_file}"
    if os.path.exists(cache_path) & args.new_spread==False:
        print("Loading cached sets...")
        trainset = torch.load(f"cache/trainset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt",weights_only=False)
        validset = torch.load(f"cache/validset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt",weights_only=False)
        testset = torch.load(f"cache/testset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt",weights_only=False)
        calibset = torch.load(f"cache/calibset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt",weights_only=False)
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
        dataset = DT2Dataset(dt2_file=dt2_file, include_elevation_in_features=False, normalize=args.normalize_elev)
        print("dataset exists!")
        # Resample:
        total = dataset.coords.shape[0]
        #print(f"min of coord: ({min(dataset.coords[:,0])},{min(dataset.coords[:,1])})")
        #print(f"max of coord: ({max(dataset.coords[:,0])},{max(dataset.coords[:,1])})")
        keep_n = int(total * args.keep_n)
        num_total_dataset = keep_n 
        num_valid = int(args.validation_size * num_total_dataset)
        num_calib = int(args.calib_percentage*num_total_dataset)
        num_test = int(args.validation_size * num_total_dataset)
        num_train = num_total_dataset - num_valid -num_test - num_calib
        #keep_n = int(total * 0.2)
        #Normal: 
        if(args.datasampling == 'uniform' and args.setsdistribtuion=='equal'):
            selected_idx = np.random.RandomState(seed=args.random_seed).choice(total, size=num_total_dataset, replace=False)
            dataset.coords = dataset.coords[selected_idx]
            dataset.features = dataset.features[selected_idx]
            dataset.y = dataset.y[selected_idx]

            # Split Indices
            # Random shuffle and split
            perm = np.random.RandomState(seed=args.random_seed).permutation(dataset.coords.shape[0])
            selected_idx_train = perm[:num_train]
            selected_idx_val = perm[num_train:num_train + num_valid]
            selected_idx_test = perm[num_train + num_valid:num_train + num_valid + num_test]
            selected_idx_calib = perm[num_train + num_valid + num_test:]
            #trainset, validset, testset = sets_creation_func_equal(dataset, selected_idx, num_total_dataset,args)
            trainset, validset, testset, calibset = sets_creation_func(dataset, selected_idx_train, selected_idx_val, selected_idx_test, selected_idx_calib)
        elif(args.datasampling == 'normal' and args.setsdistribtuion=='equal'):
            # # Extract coordinate bounds
            # lat_min, lat_max = dataset.coords[:, 0].min().item(), dataset.coords[:, 0].max().item()
            # lon_min, lon_max = dataset.coords[:, 1].min().item(), dataset.coords[:, 1].max().item()

            # # Random center within map bounds
            # rng = np.random.RandomState(seed=args.random_seed)
            # random_center = np.array([
            #     rng.uniform(lat_min, lat_max),
            #     rng.uniform(lon_min, lon_max)
            # ])
            # cov = np.diag([0.01, 0.01])  # adjust spread of Gaussian as needed

            # # Convert coords to NumPy
            # coords_np = dataset.coords.numpy()
            # # Compute probability of each point under the Gaussian
            # prob_density = multivariate_normal(mean=random_center, cov=cov).pdf(coords_np)

            # # Normalize to sum to 1 for sampling
            # prob_density /= prob_density.sum()

            # # Sample based on these probabilities
            # selected_idx = rng.choice(len(coords_np), size=keep_n, replace=False, p=prob_density)
            # # Optional: print center
            # print(f"[Gaussian Sampling] Center = ({random_center[0]:.4f}, {random_center[1]:.4f})")
            selected_idx = selected_ind_normal(dataset,0,keep_n)
            dataset.coords = dataset.coords[selected_idx]
            dataset.features = dataset.features[selected_idx]
            dataset.y = dataset.y[selected_idx]
            # Split into train and test sets
            num_total_train = int(dataset.coords.shape[0] * 0.8)  # Use 80% for training
            # Random shuffle and split
            perm = np.random.RandomState(seed=args.random_seed).permutation(dataset.coords.shape[0])
            trainset, validset, testset = sets_creation_func_equal(dataset, selected_idx,args)


        elif(args.datasampling == 'normal' and args.setsdistribtuion=='diff'):
            # CASE 1: Val, Test are mu_train+mu
            num_valid = int(args.validation_size * num_total_dataset)
            num_calib = int(args.calib_percentage*num_total_dataset)
            num_test = int(args.validation_size * num_total_dataset)
            num_train = num_total_dataset - num_valid -num_test - num_calib
            selected_idx_train = selected_ind_normal(dataset, mu=0, size=num_train, args=args)
            selected_idx_val = selected_ind_normal(dataset, mu=args.sampling_mu, size=num_valid, args=args, exclude_idx=selected_idx_train)
            selected_idx_test = selected_ind_normal(dataset, mu=args.sampling_mu, size=num_test, args=args, exclude_idx=np.concatenate([selected_idx_train, selected_idx_val]))
            selected_idx_calib = selected_ind_normal(dataset, mu=args.sampling_mu, size=num_calib, args=args, exclude_idx=np.concatenate([selected_idx_train, selected_idx_val,selected_idx_test]))

            trainset, validset, testset, calibset = sets_creation_func(dataset, selected_idx_train, selected_idx_val, selected_idx_test, selected_idx_calib)
        # Calibration Set Creation
        num_total_train = len(trainset.y)
        num_calib = int(args.calib_percentage*num_total_train)
        num_train = num_total_train - num_calib
        print(f"num_total_train: {num_total_train}, num_calib: {num_calib}, num_train:{num_train}")
        # Split trainset into train + calibration
        train_coords, train_features, train_y = trainset.coords[:num_train], trainset.features[:num_train], trainset.y[:num_train]
        calib_coords, calib_features, calib_y = trainset.coords[num_train:], trainset.features[num_train:], trainset.y[num_train:]

        trainset =  SpatialDataset(
                coords=train_coords.numpy(),
                features=train_features.numpy(),
                y=train_y.numpy()
            )
        
        calibset =  SpatialDataset(
            coords=calib_coords.numpy(),
            features=calib_features.numpy(),
            y=calib_y.numpy()
        )
        # Log Range:
        # min_y = torch.min(trainset.y)
        # trainset.y= torch.log(trainset.y+min_y)
        # testset.y = torch.log(testset.y+min_y)
        # validset.y = torch.log(validset.y+min_y)

        # print(f"NEW trainset range with log: [{torch.min(trainset.y), torch.max(trainset.y)}]")
        # print(f"NEW valset range with log: [{torch.min(validset.y), torch.max(validset.y)}]")
        print("NEW NORM IS COMING!")
        y_mean = trainset.y[:num_train].mean(dim=0, keepdim=True)
        y_std = trainset.y[:num_train].std(dim=0, keepdim=True) + 1e-6
        trainset.y = (trainset.y - y_mean) / y_std
        validset.y = (validset.y - y_mean) / y_std
        trainset.y_mean = y_mean # CHECK THIS WRITING
        trainset.y_std = y_std
        testset.y = (testset.y - y_mean) / y_std 
        calibset.y = (calibset.y - y_mean) / y_std 
        
        inspect_dataset(trainset, name="Train")
        inspect_dataset(testset, name="Test")
        torch.save(trainset, f"cache/trainset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt")
        torch.save(testset, f"cache/validset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt")
        torch.save(testset, f"cache/testset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt")
        torch.save(calibset, f"cache/calibset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt")
        #set_plot_2(trainset)
        #set_plot_2(testset)
        # Feature normalization is already handled in DT2Dataset, so no need to repeat
    return trainset, validset, testset, calibset

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

def selected_ind_normal(dataset,mu,size,args,exclude_idx=None):
    # Extract coordinate bounds
    lat_min, lat_max = dataset.coords[:, 0].min().item(), dataset.coords[:, 0].max().item()
    lon_min, lon_max = dataset.coords[:, 1].min().item(), dataset.coords[:, 1].max().item()

    # Random center within map bounds
    rng = np.random.RandomState(seed=args.random_seed)
    # random_center = np.array([
    #     rng.uniform(lat_min, lat_max),
    #     rng.uniform(lon_min, lon_max)
    # ])


    center = np.array([(lat_max + lat_min) / 2, (lon_max + lon_min) / 2])
    mean = center + mu
    cov = np.diag([0.01, 0.01])

    # Convert coords to NumPy
    coords_np = dataset.coords.numpy()
    # # Compute probability of each point under the Gaussian
    # prob_density = multivariate_normal(mean=random_center+mu, cov=cov).pdf(coords_np)

    # # Normalize to sum to 1 for sampling
    # prob_density /= prob_density.sum()

    # # Sample based on these probabilities
    # selected_idx = rng.choice(len(coords_np), size=size, replace=False, p=prob_density)
     # Mask out already used indices
    all_indices = np.arange(len(coords_np))
    if exclude_idx is not None:
        mask = np.ones(len(coords_np), dtype=bool)
        mask[exclude_idx] = False
        coords_np = coords_np[mask]
        all_indices = all_indices[mask]

    # Gaussian PDF
    prob_density = multivariate_normal(mean=mean, cov=cov).pdf(coords_np)
    prob_density /= prob_density.sum()

    selected_local = rng.choice(len(coords_np), size=size, replace=False, p=prob_density)
    selected_idx = all_indices[selected_local]
    return selected_idx


def sets_creation_func_equal(dataset, selected_idx, args):   

    dataset.coords = dataset.coords[selected_idx]
    dataset.features = dataset.features[selected_idx]
    dataset.y = dataset.y[selected_idx]
    # Split into train and test sets
    num_total_train = int(dataset.coords.shape[0] * 0.8)  # Use 80% for training
    # Random shuffle and split
    perm = np.random.RandomState(seed=args.random_seed).permutation(dataset.coords.shape[0])

    testset = SpatialDataset(
        coords=dataset.coords[perm[num_total_train:]].numpy(),
        features=dataset.features[perm[num_total_train:]].numpy(),
        y=dataset.y[perm[num_total_train:]].numpy()
    )

    #if args.normalize_elev:
    num_valid = args.validation_size * num_total_train
    num_train = int(num_total_train - num_valid)

    trainset = SpatialDataset(
        coords=dataset.coords[:num_train].numpy(),
        features=dataset.features[:num_train].numpy(),
        y=dataset.y[:num_train].numpy()
    )

    validset = SpatialDataset(
        coords=dataset.coords[num_train:].numpy(),
        features=dataset.features[num_train:].numpy(),
        y=dataset.y[num_train:].numpy()
    )

    return trainset, validset, testset


def sets_creation_func(dataset, selected_idx_train, selected_idx_val, selected_idx_test, selected_idx_calib):   
    testset = SpatialDataset(
    coords=dataset.coords[selected_idx_test].numpy(),
    features=dataset.features[selected_idx_test].numpy(),
    y=dataset.y[selected_idx_test].numpy()
    )

    trainset = SpatialDataset(
        coords=dataset.coords[selected_idx_train].numpy(),
        features=dataset.features[selected_idx_train].numpy(),
        y=dataset.y[selected_idx_train].numpy()
    )

    validset = SpatialDataset(
        coords=dataset.coords[selected_idx_val].numpy(),
        features=dataset.features[selected_idx_val].numpy(),
        y=dataset.y[selected_idx_val].numpy()
    )

    calibset = SpatialDataset(
        coords=dataset.coords[selected_idx_calib].numpy(),
        features=dataset.features[selected_idx_calib].numpy(),
        y=dataset.y[selected_idx_calib].numpy()
    )

    return trainset, validset, testset, calibset
