import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
from data import SpatialDataset
import os
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

# =====================================
# DT2Dataset: reads a single .tiff/.dt2 tile
# =====================================
class DT2Dataset(Dataset):
    """Dataset for DTED elevation maps in SpatialDataset format."""
    def __init__(self, dt2_file, include_elevation_in_features=False, normalize=True):
        """
        Args:
            dt2_file: path to the .tiff/.dt2 file
            include_elevation_in_features: if True, adds elevation as part of the feature vector
            normalize: (currently reserved) if True, normalize features (coords normalization handled downstream)
        Notes:
            transform maps pixel (row,col) -> geographic (lon,lat):
            (pixel_width, row_rot, x_min, col_rot, pixel_height, y_max)
        """
        with rasterio.open(dt2_file) as src:
            print("I am reading!")
            elevation = src.read(1)  # (H, W)
            transform = src.transform
            height, width = elevation.shape

            # Coordinate grid (lon per column, lat per row)
            lon_coords = np.array([transform[2] + i * transform[0] for i in range(width)])
            lat_coords = np.array([transform[5] + j * transform[4] for j in range(height)])
            lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

            # Flatten
            coords = np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=1)  # [N,2] (lat,lon)
            elevations = elevation.flatten().astype(np.float32).reshape(-1, 1)   # [N,1]

            # Features: by default just coords; optionally concat elevation
            if include_elevation_in_features:
                features = np.concatenate([coords, elevations], axis=1)
            else:
                features = coords

            # Labels
            y = elevations

            # Tensors
            self.coords = torch.from_numpy(coords).float()
            self.features = torch.from_numpy(features).float()
            self.y = torch.from_numpy(y).float()
            

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
         return self.coords[idx], self.features[idx], self.y[idx]
    # def __getitem__(self, idx):
    #     # existing loading of coords, features, y
    #     coords = self.coords[idx] # [N,2] in lat,lon
    #     feats = self.features[idx] # [N,F] or []
    #     y = self.y[idx] # [N]
    #     # Normalize coords to [0,1] (min-max per tile)
    #     c_min = coords.min(dim=0).values
    #     c_max = coords.max(dim=0).values
    #     denom = torch.clamp(c_max - c_min, min=1e-6)
    #     coords01 = (coords - c_min) / denom
    #     # Build masks; if you already have train/test point splits, map them to obs/query
    #     if hasattr(self, 'observed_indices') and hasattr(self,'query_indices'):
    #         obs_mask = torch.zeros(coords.shape[0], dtype=torch.bool)
    #         qry_mask = torch.zeros(coords.shape[0], dtype=torch.bool)
    #         obs_mask[self.observed_indices[idx]] = True
    #         qry_mask[self.query_indices[idx]] = True
    #     else:# Fallback: random split 80/20 at load time (only if you don’t already provide)
    #         N = coords.shape[0]
    #         k = max(1, int(0.8 * N))
    #         perm = torch.randperm(N)
    #         obs_idx = perm[:k]
    #         qry_idx = perm[k:]
    #         obs_mask = torch.zeros(N, dtype=torch.bool); obs_mask[obs_idx] = True
    #         qry_mask = torch.zeros(N, dtype=torch.bool); qry_mask[qry_idx] = True
    #     sample = {'coords': coords01.float(), 
    #               'features': feats.float() if feats.numel() > 0 else None, 
    #               'y': y.float(),
    #               'obs_mask': obs_mask,
    #               'query_mask': qry_mask,
    #               # if you use padding in collation, also compute pad_mask there
    #               }
    #     return sample
    

# =====================================
# Helper plotting / inspection
# =====================================

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

    coords = dataset.coords.numpy()
    print(f"➤ Lat range: {coords[:, 0].min():.4f} - {coords[:, 0].max():.4f}")
    print(f"➤ Lon range: {coords[:, 1].min():.4f} - {coords[:, 1].max():.4f}")


def set_plot(dataset):
    extent = [dataset.coords[:, 1].min(), dataset.coords[:, 1].max(),
              dataset.coords[:, 0].min(), dataset.coords[:, 0].max()]
    plt.figure(figsize=(10, 8))
    # dataset.y is 1D (N,1); imshow expects 2D grid — keep for quick-look only if you reshape externally
    plt.scatter(dataset.coords[:, 1], dataset.coords[:, 0], c=dataset.y.squeeze(-1), s=2, cmap="terrain")
    plt.colorbar(label="Elevation (m)")
    plt.title("DTED Level 2 Elevation (scatter)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.show()


def set_plot_2(dataset):
    plt.figure(figsize=(10, 8))
    extent = [dataset.coords[:, 1].min(), dataset.coords[:, 1].max(),
              dataset.coords[:, 0].min(), dataset.coords[:, 0].max()]
    plt.scatter(dataset.coords[:, 1], dataset.coords[:, 0], c=dataset.y.squeeze(-1), s=2, cmap="terrain")
    plt.colorbar(label="Elevation (m)")
    lats = dataset.coords[:, 0]
    lons = dataset.coords[:, 1]
    plt.scatter(lons, lats, s=2, c='red', label='Points', alpha=0.6)
    plt.title("DTED Level 2 with Sampled Points")
    plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.legend(); plt.show()

# =====================================
# Sampling utils
# =====================================

def selected_ind_normal(dataset, mu, size, args, exclude_idx=None):
    lat_min, lat_max = dataset.coords[:, 0].min().item(), dataset.coords[:, 0].max().item()
    lon_min, lon_max = dataset.coords[:, 1].min().item(), dataset.coords[:, 1].max().item()

    rng = np.random.RandomState(seed=args.random_seed)
    center = np.array([(lat_max + lat_min) / 2, (lon_max + lon_min) / 2])
    mean = center + mu
    cov = np.diag([0.01, 0.01])

    coords_np = dataset.coords.numpy()

    # Exclude indices if provided
    all_indices = np.arange(len(coords_np))
    if exclude_idx is not None:
        mask = np.ones(len(coords_np), dtype=bool)
        mask[exclude_idx] = False
        coords_np = coords_np[mask]
        all_indices = all_indices[mask]

    prob_density = multivariate_normal(mean=mean, cov=cov).pdf(coords_np)
    prob_density /= prob_density.sum()

    selected_local = rng.choice(len(coords_np), size=size, replace=False, p=prob_density)
    selected_idx = all_indices[selected_local]
    return selected_idx


def sets_creation_func(dataset, selected_idx_train, selected_idx_val, selected_idx_test, selected_idx_calib):
    testset = SpatialDataset(
        coords=dataset.coords[selected_idx_test].numpy(),
        features=dataset.features[selected_idx_test].numpy(),
        y=dataset.y[selected_idx_test].numpy(),
        #obs_mask: dataset.y[selected_idx_test].numpy(),
        #query_mask: dataset.y[selected_idx_test].numpy()
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

# =====================================
# Main loader with 4-way split (train/valid/test/calib)
# =====================================

def load_dt2_data(args):
    """
    Load data for training, validation, test, and calibration from a DTED file.

    Returns
    -------
    trainset, validset, testset, calibset : SpatialDataset objects
    """
    # file path
    os.makedirs("cache/", exist_ok=True)
    cache_key = f"{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}"
    dt2_file = os.path.join(args.data_path, args.dataset + ".tiff")
    print(f"[DEBUG] Using dt2_file path: {dt2_file}")
    assert os.path.isfile(dt2_file), f"File does not exist: {dt2_file}"

    cache_exists = (
        os.path.exists(f"cache/trainset_{cache_key}.pt") and
        os.path.exists(f"cache/validset_{cache_key}.pt") and
        os.path.exists(f"cache/testset_{cache_key}.pt") and
        os.path.exists(f"cache/calibset_{cache_key}.pt")
    )

    if cache_exists and (args.new_spread == False):
        print("Loading cached sets...")
        trainset = torch.load(f"cache/trainset_{cache_key}.pt", weights_only=False)
        validset = torch.load(f"cache/validset_{cache_key}.pt", weights_only=False)
        testset  = torch.load(f"cache/testset_{cache_key}.pt",  weights_only=False)
        calibset = torch.load(f"cache/calibset_{cache_key}.pt", weights_only=False)
        return trainset, validset, testset, calibset

    print("Creating and caching sets...")
    dataset = DT2Dataset(dt2_file=dt2_file, include_elevation_in_features=False, normalize=getattr(args, 'normalize_elev', False))
    print("dataset exists!")

    # ------- Resample point subset (deterministic) -------
    total = dataset.coords.shape[0]
    keep_n = int(total * args.keep_n)
    rng = np.random.RandomState(seed=args.random_seed)
    selected_idx = rng.choice(total, size=keep_n, replace=False) if args.datasampling == 'uniform' else None

    num_total_dataset = keep_n
    # use the SAME ratio for val/test/calib, as in your code
    num_valid = int(args.validation_size * num_total_dataset)
    num_calib = int(args.validation_size * num_total_dataset)
    num_test  = int(args.validation_size * num_total_dataset)
    num_train = num_total_dataset - num_valid - num_test - num_calib
    assert num_train > 0, "Non-positive train size; reduce validation_size or keep_n."

    if (args.datasampling == 'uniform' and args.setsdistribtuion == 'equal'):
        perm = rng.permutation(len(selected_idx))
        sel = selected_idx
        idx_tr = sel[perm[:num_train]]
        idx_va = sel[perm[num_train:num_train + num_valid]]
        idx_te = sel[perm[num_train + num_valid:num_train + num_valid + num_test]]
        idx_ca = sel[perm[num_train + num_valid + num_test:]]
        trainset, validset, testset, calibset = sets_creation_func(dataset, idx_tr, idx_va, idx_te, idx_ca)

    elif (args.datasampling == 'normal' and args.setsdistribtuion == 'equal'):
        sel = selected_ind_normal(dataset, 0, keep_n, args)
        perm = rng.permutation(len(sel))
        idx_tr = sel[perm[:num_train]]
        idx_va = sel[perm[num_train:num_train + num_valid]]
        idx_te = sel[perm[num_train + num_valid:num_train + num_valid + num_test]]
        idx_ca = sel[perm[num_train + num_valid + num_test:]]
        trainset, validset, testset, calibset = sets_creation_func(dataset, idx_tr, idx_va, idx_te, idx_ca)

    elif (args.datasampling == 'normal' and args.setsdistribtuion == 'diff'):
        idx_tr = selected_ind_normal(dataset, mu=0, size=num_train, args=args)
        idx_va = selected_ind_normal(dataset, mu=args.sampling_mu, size=num_valid, args=args, exclude_idx=idx_tr)
        idx_te = selected_ind_normal(dataset, mu=args.sampling_mu, size=num_test, args=args, exclude_idx=np.concatenate([idx_tr, idx_va]))
        idx_ca = selected_ind_normal(dataset, mu=args.sampling_mu, size=num_calib, args=args, exclude_idx=np.concatenate([idx_tr, idx_va, idx_te]))
        trainset, validset, testset, calibset = sets_creation_func(dataset, idx_tr, idx_va, idx_te, idx_ca)

    else:
        raise ValueError("Unsupported combination for datasampling/setsdistribtuion")

    print(f"num_total: {num_total_dataset}, num_train: {num_train}, num_val: {num_valid}, num_test: {num_test}, num_calib: {num_calib}")

    # ------- Normalize targets by train statistics -------
    y_mean = trainset.y.mean(dim=0, keepdim=True)
    y_std  = trainset.y.std(dim=0, keepdim=True) + 1e-6

    trainset.y = (trainset.y - y_mean) / y_std
    validset.y = (validset.y - y_mean) / y_std
    testset.y  = (testset.y  - y_mean) / y_std
    calibset.y = (calibset.y - y_mean) / y_std

    # Keep for inverse-transform if needed
    trainset.y_mean = y_mean
    trainset.y_std  = y_std

    # ------- Inspect & Cache -------
    inspect_dataset(trainset, name="Train")
    inspect_dataset(testset, name="Test")

    torch.save(trainset, f"cache/trainset_{cache_key}.pt")
    torch.save(validset, f"cache/validset_{cache_key}.pt")  # (fix) save validset correctly
    torch.save(testset,  f"cache/testset_{cache_key}.pt")
    torch.save(calibset, f"cache/calibset_{cache_key}.pt")

    return trainset, validset, testset, calibset

# import numpy as np
# import torch
# import rasterio
# from torch.utils.data import Dataset
# from data import SpatialDataset
# import os
# from matplotlib import pyplot as plt
# from scipy.stats import multivariate_normal
# class DT2Dataset(Dataset):
#     """Dataset for DTED elevation maps in SpatialDataset format."""
    
#     def __init__(self, dt2_file, include_elevation_in_features=False, normalize=True):
#         """
#         Args:
#             dt2_file: path to the .dt2 file
#             include_coords_in_features: if True, adds lat/lon as part of the feature vector
#             normalize: if True, normalize the feature values
            
#             Notes:
#             transform is a 2D affine transformation that maps pixel coordinates (row, col) to geographic coordinates (lon, lat).
#             transform = (pixel_width, row_rotation, x_min, col_rotation, pixel_height, y_max).
#             transform[0] is pixel width in degrees (Δlon)
#             transform[2] is lon_min (start longitude)
#             transform[4] is pixel height (negative, because images start from top-left)
#             transform[5] is lat_max (top latitude)

#         """
#         with rasterio.open(dt2_file) as src:
#             print("I am reading!")
#             elevation = src.read(1)  # shape: (height, width). elevation[i,j] - gives the elevation in meters at (i,j)
#             transform = src.transform
#             height, width = elevation.shape

#             # Create coordinate grid
#             lon_coords = np.array([transform[2] + i * transform[0] for i in range(width)]) # 1D array of longitudes for each column
#             lat_coords = np.array([transform[5] + j * transform[4] for j in range(height)]) # 1D array of latitudes for each row
#             lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords) # Creates full grids of shape (height, width) — so now each pixel has an exact lat-lon pair.

#             # Flatten all arrays
#             coords = np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=1)  # [n, 2]
#             elevations = elevation.flatten().astype(np.float32).reshape(-1, 1)    # [n, 1]

#             # Features: only coords by default or coords and elevation
#             '''
#             if include_elevation_in_features:
#                 features = np.concatenate([coords, elevations], axis=1)
#                 print(f"Features shape:{features.shape}")
#             else:
#                 features = coords
#             '''
#             features = coords
#             # Labels
#             y = elevations  
                
#             # Convert to tensors
#             self.coords = torch.from_numpy(coords).float()
#             self.features = torch.from_numpy(features).float()
#             self.y = torch.from_numpy(y).float()
            
#             #if normalize: #CHECK: Should I normalize the coords?
#             #    self._normalize_features()

#     def _normalize_features(self):
#         self.feature_mean = self.features.mean(dim=0, keepdim=True)
#         self.feature_std = self.features.std(dim=0, keepdim=True) + 1e-6
#         self.features = (self.features - self.feature_mean) / self.feature_std

#     def __len__(self):
#         return self.coords.shape[0]

#     def __getitem__(self, idx):
#         return self.coords[idx], self.features[idx], self.y[idx]

# def load_dt2_data(args):
#     """
#     Load data for training and testing from DT2 (elevation) files

#     Args
#     ----
#     args : will use three fields, args.dataset, args.data_path, args.random_seed  

#     Returns
#     -------
#     coords    : np.ndarray, shape (N, 2), coordinates of the data points
#     features  : np.ndarray, shape (N, D), features of the data points
#     y         : np.ndarray, shape (N, 1), labels of the data points
#     num_total_train : int, number of training data points. The first `num_total_train` 
#                       of instances from three other return values should form the training set
#     """
#     # data file path
#     os.makedirs("cache/", exist_ok=True)
#     cache_path = f"cache/trainset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt"
#     dt2_file = os.path.join(args.data_path, args.dataset + ".tiff")
#     print(f"[DEBUG] Using dt2_file path: {dt2_file}")
#     assert os.path.isfile(dt2_file), f"File does not exist: {dt2_file}"
#     if (os.path.exists(cache_path)) and (args.new_spread==False):
#         print("Loading cached sets...")
#         trainset = torch.load(f"cache/trainset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt",weights_only=False)
#         validset = torch.load(f"cache/validset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt",weights_only=False)
#         testset = torch.load(f"cache/testset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt",weights_only=False)
#         calibset = torch.load(f"cache/calibset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt",weights_only=False)
#     else:
#         print("Creating and caching sets...")

#         dataset = DT2Dataset(dt2_file=dt2_file, include_elevation_in_features=False, normalize=args.normalize_elev)
#         print("dataset exists!")
#         # Resample:
#         total = dataset.coords.shape[0]
#         keep_n = int(total * args.keep_n)
#         num_total_dataset = keep_n 
#         num_valid = int(args.validation_size * num_total_dataset)
#         num_calib = int(args.validation_size * num_total_dataset)
#         num_test = int(args.validation_size * num_total_dataset)
#         num_train = num_total_dataset - num_valid -num_test - num_calib

#         if(args.datasampling == 'uniform' and args.setsdistribtuion=='equal'):

#             selected_idx = np.random.RandomState(seed=args.random_seed).choice(total, size=num_total_dataset, replace=False)

#             # Split Indices
#             perm = np.random.RandomState(seed=args.random_seed).permutation(len(selected_idx))
#             selected_idx_train = selected_idx[perm[:num_train]]
#             selected_idx_val = selected_idx[perm[num_train:num_train + num_valid]]
#             selected_idx_test = selected_idx[perm[num_train + num_valid:num_train + num_valid + num_test]]
#             selected_idx_calib = selected_idx[perm[num_train + num_valid + num_test:]]

#             # Splits Sets
#             trainset, validset, testset, calibset = sets_creation_func(dataset, selected_idx_train, selected_idx_val, selected_idx_test, selected_idx_calib)

#         elif(args.datasampling == 'normal' and args.setsdistribtuion=='equal'):
            
#             selected_idx = selected_ind_normal(dataset,0,keep_n,args)

#             # Split Indices
#             perm = np.random.RandomState(seed=args.random_seed).permutation(len(selected_idx))
#             selected_idx_train = selected_idx[perm[:num_train]]
#             selected_idx_val = selected_idx[perm[num_train:num_train + num_valid]]
#             selected_idx_test = selected_idx[perm[num_train + num_valid:num_train + num_valid + num_test]]
#             selected_idx_calib = selected_idx[perm[num_train + num_valid + num_test:]]
 
#             trainset, validset, testset, calibset = sets_creation_func(dataset, selected_idx_train, selected_idx_val, selected_idx_test, selected_idx_calib)


#         elif(args.datasampling == 'normal' and args.setsdistribtuion=='diff'):

#             # CASE 1: Val, Test, Calib are mu_train+mu
#             selected_idx_train = selected_ind_normal(dataset, mu=0, size=num_train, args=args)
#             selected_idx_val = selected_ind_normal(dataset, mu=args.sampling_mu, size=num_valid, args=args, exclude_idx=selected_idx_train)
#             selected_idx_test = selected_ind_normal(dataset, mu=args.sampling_mu, size=num_test, args=args, exclude_idx=np.concatenate([selected_idx_train, selected_idx_val]))
#             selected_idx_calib = selected_ind_normal(dataset, mu=args.sampling_mu, size=num_calib, args=args, exclude_idx=np.concatenate([selected_idx_train, selected_idx_val,selected_idx_test]))

#             trainset, validset, testset, calibset = sets_creation_func(dataset, selected_idx_train, selected_idx_val, selected_idx_test, selected_idx_calib)
        
#         print(f"num_total: {num_total_dataset}, num_calib: {num_calib}, num_train:{num_train}, num_val:{num_valid},  num_calib:{num_calib}")
#         print("NEW NORM IS COMING!")
        
#         y_mean = trainset.y[:num_train].mean(dim=0, keepdim=True)
#         y_std = trainset.y[:num_train].std(dim=0, keepdim=True) + 1e-6
#         trainset.y = (trainset.y - y_mean) / y_std
#         validset.y = (validset.y - y_mean) / y_std
#         trainset.y_mean = y_mean # CHECK THIS WRITING
#         trainset.y_std = y_std
#         testset.y = (testset.y - y_mean) / y_std 
#         calibset.y = (calibset.y - y_mean) / y_std 
        
#         inspect_dataset(trainset, name="Train")
#         inspect_dataset(testset, name="Test")
#         torch.save(trainset, f"cache/trainset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt")
#         torch.save(testset, f"cache/validset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt")
#         torch.save(testset, f"cache/testset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt")
#         torch.save(calibset, f"cache/calibset_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt")
#         #set_plot_2(trainset)
#         #set_plot_2(testset)
#         # Feature normalization is already handled in DT2Dataset, so no need to repeat
#     return trainset, validset, testset, calibset

# def inspect_dataset(dataset, name="Train"):

#     print(f"\n {name} Dataset Summary")
#     print(f"➤ Number of points: {len(dataset)}")
#     print(f"➤ Coords shape: {dataset.coords.shape}")
#     print(f"➤ Feature shape: {dataset.features.shape}")
#     print(f"➤ Label shape: {dataset.y.shape}")
#     print(f"➤ Feature mean/std (first 5 dims):")
#     print(f"   mu = {dataset.features.mean(0)[:5].numpy()}")
#     print(f"  std = {dataset.features.std(0)[:5].numpy()}")
#     print(f"➤ Elevation min/max: {dataset.y.min().item():.2f} / {dataset.y.max().item():.2f}")

#     # Coordinates info
#     coords = dataset.coords.numpy()
#     print(f"➤ Lat range: {coords[:, 0].min():.4f} - {coords[:, 0].max():.4f}")
#     print(f"➤ Lon range: {coords[:, 1].min():.4f} - {coords[:, 1].max():.4f}")

# def set_plot(dataset):
#     extent = [dataset.coords[1].min(),dataset.coords[1].max(), dataset.coords[0].min(), dataset.coords[0].max()]  # Get geographical extent
#     plt.figure(figsize=(10, 8))
#     plt.imshow(dataset.y, cmap="terrain", extent=extent, origin="upper")
#     plt.colorbar(label="Elevation (m)")
#     plt.title("DTED Level 2 Elevation Data")
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.show()
    
# def set_plot_2(dataset):
#     plt.figure(figsize=(10, 8))
#     extent = [dataset.coords[:, 1].min(),dataset.coords[:, 1].max(), dataset.coords[:, 0].min(), dataset.coords[:,0].max()]  # Get geographical extent
#     plt.imshow(dataset.y, cmap="terrain", extent=extent, origin="upper")
#     plt.colorbar(label="Elevation (m)")
    
#     # Scatter your training points on top
#     lats = dataset.coords[:, 0]
#     lons = dataset.coords[:, 1]
#     plt.scatter(lons, lats, s=2, c='red', label='Training points', alpha=0.6)

#     plt.title("DTED Level 2 with Sampled Training Points")
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.legend()
#     plt.show()

# def selected_ind_normal(dataset,mu,size,args,exclude_idx=None):
#     # Extract coordinate bounds
#     lat_min, lat_max = dataset.coords[:, 0].min().item(), dataset.coords[:, 0].max().item()
#     lon_min, lon_max = dataset.coords[:, 1].min().item(), dataset.coords[:, 1].max().item()

#     # Random center within map bounds
#     rng = np.random.RandomState(seed=args.random_seed)
#     # random_center = np.array([
#     #     rng.uniform(lat_min, lat_max),
#     #     rng.uniform(lon_min, lon_max)
#     # ])

#     center = np.array([(lat_max + lat_min) / 2, (lon_max + lon_min) / 2])
#     mean = center + mu
#     cov = np.diag([0.01, 0.01])

#     # Convert coords to NumPy
#     coords_np = dataset.coords.numpy()

#      # Mask out already used indices
#     all_indices = np.arange(len(coords_np))
#     if exclude_idx is not None:
#         mask = np.ones(len(coords_np), dtype=bool)
#         mask[exclude_idx] = False
#         coords_np = coords_np[mask]
#         all_indices = all_indices[mask]

#     # Gaussian PDF
#     prob_density = multivariate_normal(mean=mean, cov=cov).pdf(coords_np)
#     prob_density /= prob_density.sum()

#     selected_local = rng.choice(len(coords_np), size=size, replace=False, p=prob_density)
#     selected_idx = all_indices[selected_local]

#     return selected_idx



# def sets_creation_func(dataset, selected_idx_train, selected_idx_val, selected_idx_test, selected_idx_calib):   
#     testset = SpatialDataset(
#     coords=dataset.coords[selected_idx_test].numpy(),
#     features=dataset.features[selected_idx_test].numpy(),
#     y=dataset.y[selected_idx_test].numpy()
#     )

#     trainset = SpatialDataset(
#         coords=dataset.coords[selected_idx_train].numpy(),
#         features=dataset.features[selected_idx_train].numpy(),
#         y=dataset.y[selected_idx_train].numpy()
#     )

#     validset = SpatialDataset(
#         coords=dataset.coords[selected_idx_val].numpy(),
#         features=dataset.features[selected_idx_val].numpy(),
#         y=dataset.y[selected_idx_val].numpy()
#     )

#     calibset = SpatialDataset(
#         coords=dataset.coords[selected_idx_calib].numpy(),
#         features=dataset.features[selected_idx_calib].numpy(),
#         y=dataset.y[selected_idx_calib].numpy()
#     )

#     return trainset, validset, testset, calibset

def collate_setformer(batch):
    # batch: list of dicts with keys as in __getitem__
    B = len(batch)
    Nmax = max(item['coords01'].shape[0] for item in batch)
    def pad_tensor(t, pad_dim, value=0.0):
        if t is None:
            return None
        out = t.new_full((B, Nmax, t.shape[-1]), value)
        for i, it in enumerate(batch):
            n = it['coords01'].shape[0]
            out[i, :n] = t[i] if t.dim()==3 else t[:n]
        return out
    coords = torch.zeros(B, Nmax, 2)
    feats_list = []
    y = torch.zeros(B, Nmax)
    obs = torch.zeros(B, Nmax, dtype=torch.bool)
    qry = torch.zeros(B, Nmax, dtype=torch.bool)
    pad = torch.ones(B, Nmax, dtype=torch.bool)  # True = padded
    for i, it in enumerate(batch):
        n = it['coords01'].shape[0]
        coords[i, :n] = it['coords01']
        if it['feats'] is not None:
            feats_list.append(it['feats'])
        y[i, :n] = it['y']
        obs[i, :n] = it['obs_mask']
        qry[i, :n] = it['query_mask']
        pad[i, :n] = False
    feats = None
    if len(feats_list) == B:  # only if everyone has feats
        F = feats_list[0].shape[-1]
        feats = torch.zeros(B, Nmax, F)
        for i, it in enumerate(batch):
            n = it['coords01'].shape[0]
            feats[i, :n] = it['feats']
    return coords, feats, y, obs, qry, pad
