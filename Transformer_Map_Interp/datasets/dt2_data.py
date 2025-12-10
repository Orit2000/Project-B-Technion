import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
from Transformer_Map_Interp.datasets.data import SpatialDataset
import os
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.neighbors import KDTree
import time
import pandas as pd
import torch

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
            # nodata = src.nodata
            # elevation = np.ma.masked_equal(elevation, nodata)
            
            # valid = ~elevation.mask                     # boolean mask of kept pixels
            # elevation = elevation[valid]                     # elevations at those pixels (1D)
            
            # Coordinate grid (lon per column, lat per row)
            lon_coords = np.array([transform[2] + i * transform[0] for i in range(width)])
            lat_coords = np.array([transform[5] + j * transform[4] for j in range(height)])
            lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
            ### CHANGE
            valid_mask = elevation > 0
            # Calculate stats for debugging
            total_pixels = elevation.size
            kept_pixels = np.sum(valid_mask)
            print(f"Filtering: Kept {kept_pixels}/{total_pixels} points (dropped <= 0)")

            if kept_pixels == 0:
                raise ValueError(f"No valid points found in {dt2_file} with criterion > 0")

            # Apply mask to extract only valid data
            # Numpy boolean indexing automatically flattens the result into 1D arrays
            valid_lats = lat_grid[valid_mask]
            valid_lons = lon_grid[valid_mask]
            valid_elevs = elevation[valid_mask]
            # --- FILTERING LOGIC END ---

            # Stack coordinates: [N, 2] -> (lat, lon)
            coords = np.stack([valid_lats.flatten(), valid_lons.flatten()], axis=1)
            
            # Reshape elevations: [N, 1]
            elevations = valid_elevs.flatten().astype(np.float32).reshape(-1, 1)

            # Flatten
            # coords = np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=1)  # [N,2] (lat,lon)
            # elevations = elevation.flatten().astype(np.float32).reshape(-1, 1)   # [N,1]
            ### CHANGE ENDS


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


def sets_creation_func(dataset, selected_idx_train, selected_idx_val, selected_idx_test, selected_idx_calib, max_radius_km):
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

    # labels stats (scalars)
    train_y_mean = torch.as_tensor(trainset.y.mean(), dtype=torch.float32)
    train_y_std  = torch.as_tensor(trainset.y.std(),  dtype=torch.float32).clamp_min(1e-6)

    trainset.y_mean = train_y_mean
    trainset.y_std  = train_y_std
    
    train_lat_std = trainset.coords[:,0].std().float().clamp_min(1e-6)
    train_lon_std = trainset.coords[:,1].std().float().clamp_min(1e-6)
    trainset.lat_std = train_lat_std
    trainset.lon_std = train_lon_std
    print("In sets creation - before faster_add_transformer_masks")
    faster_add_transformer_masks(trainset, trainset.coords, trainset.y, trainset.y_mean, trainset.y_std,
                       max_radius_km=max_radius_km, self_exclude=True, max_obs=256, min_k=8)
    faster_add_transformer_masks(validset, trainset.coords, trainset.y, trainset.y_mean, trainset.y_std,
                         max_radius_km=max_radius_km, self_exclude=False,max_obs=256, min_k=8)
    faster_add_transformer_masks(testset, trainset.coords, trainset.y, trainset.y_mean, trainset.y_std,
                         max_radius_km=max_radius_km, self_exclude=False,max_obs=256, min_k=8)
    faster_add_transformer_masks(calibset, trainset.coords, trainset.y, trainset.y_mean, trainset.y_std,
                         max_radius_km=max_radius_km, self_exclude=False,max_obs=256, min_k=8)
    return trainset, validset, testset, calibset

def faster_add_transformer_masks_no_batching(
    dataset,
    nei_coords,
    nei_y,
    train_y_mean, 
    train_y_std,
    max_radius_km=None,
    self_exclude=True,
    max_obs=10,             # number of neighbors to keep
    coord_atol=1e-9
):
    import numpy as np
    import torch
    from sklearn.neighbors import KDTree

    ###########################################################################
    # 1) BUILD KDTree in projected (km) coordinate system
    ###########################################################################
    with torch.no_grad():
        lat_deg = nei_coords[:, 0].cpu().double().numpy()
        lon_deg = nei_coords[:, 1].cpu().double().numpy()

    lat0 = float(np.mean(lat_deg))
    k_lat = 110.574
    k_lon = 111.320 * np.cos(np.deg2rad(lat0))

    nei_xy = np.stack([lon_deg * k_lon, lat_deg * k_lat], axis=1)

    print("Building KDTree...")
    kdt = KDTree(nei_xy, metric='euclidean')
    print("KDTree built.")

    ###########################################################################
    # 2) PREPARE padded tensors
    ###########################################################################
    N = len(dataset)

    obs_coords_tensor      = torch.zeros((N, max_obs, 2), dtype=torch.float32)
    obs_y_tensor           = torch.zeros((N, max_obs), dtype=torch.float32)
    obs_coords_norm_tensor = torch.zeros((N, max_obs, 2), dtype=torch.float32)
    obs_y_norm_tensor      = torch.zeros((N, max_obs), dtype=torch.float32)

    mask_tensor = torch.zeros((N, max_obs), dtype=torch.bool)
    q_coords_tensor  = torch.zeros((N, 2), dtype=torch.float32)
    q_y_tensor       = torch.zeros((N,), dtype=torch.float32)
    q_y_norm_tensor  = torch.zeros((N,), dtype=torch.float32)

    ###########################################################################
    # 3) PREPARE query coordinates (for global radius query)
    ###########################################################################
    q_coords = dataset.coords.cpu().double().numpy()
    q_xy = np.stack([q_coords[:,1] * k_lon, q_coords[:,0] * k_lat], axis=1)

    ###########################################################################
    # 4) SINGLE GIANT radius-query (NOT batched)
    #    ❗ Warning: OOM for millions of points
    ###########################################################################
    print("Performing FULL radius query (no batching)...")
    inds_list, dists_list = kdt.query_radius(
        q_xy,
        r=max_radius_km,
        return_distance=True
    )
    print("Full radius query done.")

    ###########################################################################
    # 5) LOOP — Fill padded tensors
    ###########################################################################
    for i in range(N):
        q_coord = dataset.coords[i]
        q_y     = dataset.y[i]

        inds  = inds_list[i]
        dists = dists_list[i]

        # Self exclude
        if self_exclude:
            mask = dists > 1e-6
            inds  = inds[mask]
            dists = dists[mask]

        # Cap to max_obs nearest neighbors
        if len(inds) > max_obs:
            order = np.argsort(dists)[:max_obs]
            inds  = inds[order]
            dists = dists[order]

        # Slice neighbor data
        if len(inds) > 0:
            tinds = torch.as_tensor(inds, dtype=torch.long)
            obs_coords = nei_coords[tinds]
            obs_y      = nei_y[tinds]

            # (num,1) → (num,)
            if obs_y.ndim == 2 and obs_y.shape[1] == 1:
                obs_y = obs_y.squeeze(-1)

            num = len(inds)

            # Neighbors
            obs_coords_tensor[i, :num] = obs_coords
            obs_y_tensor[i, :num]      = obs_y
            mask_tensor[i, :num]       = True

            # Normalized neighbors
            obs_coords_norm_tensor[i, :num] = obs_coords - q_coord
            obs_y_norm_tensor[i, :num]      = (obs_y - train_y_mean) / train_y_std

        # Query point
        q_coords_tensor[i] = q_coord
        q_y_tensor[i]      = q_y
        q_y_norm_tensor[i] = (q_y - train_y_mean) / train_y_std

        if i % 50000 == 0:
            print(f"Processed {i}/{N} points")

    ###########################################################################
    # 6) Attach padded tensors to dataset
    ###########################################################################
    dataset.obs_coords      = obs_coords_tensor
    dataset.obs_y           = obs_y_tensor
    dataset.obs_coords_norm = obs_coords_norm_tensor
    dataset.obs_y_norm      = obs_y_norm_tensor
    dataset.obs_mask        = mask_tensor
    dataset.query_coords    = q_coords_tensor
    dataset.query_y         = q_y_tensor
    dataset.q_y_norm        = q_y_norm_tensor

    print("Done (non-batched version).")
    return dataset


def faster_add_transformer_masks(
    dataset,
    nei_coords,
    nei_y,
    train_y_mean, 
    train_y_std,
    max_radius_km=None,
    self_exclude=True,
    max_obs=None,         # cap number of observed tokens (optional)
    min_k=8,              # fallback K if radius leaves too few neighbors
    coord_atol=1e-9       # tolerance for coordinate equality
):

    ###########################################################################
    # 1) BUILD KDTree in projected (km) coordinate system
    ###########################################################################
    with torch.no_grad():
        nei_lat = nei_coords[:, 0].cpu().double().numpy()
        nei_lon = nei_coords[:, 1].cpu().double().numpy()

    lat0 = float(np.mean(nei_lat))
    k_lat = 110.574
    k_lon = 111.320 * np.cos(np.deg2rad(lat0))

    nei_xy = np.stack([nei_lon * k_lon, nei_lat * k_lat], axis=1)

    print("Building KDTree...")
    kdt = KDTree(nei_xy, metric='euclidean')
    print("KDTree built.")

    ###########################################################################
    # 2) PREPARE padded tensors
    ###########################################################################
    N = len(dataset)

    obs_coords_tensor = torch.zeros((N, max_obs, 2), dtype=torch.float32)
    obs_y_tensor      = torch.zeros((N, max_obs), dtype=torch.float32)
    obs_coords_norm_tensor = torch.zeros((N, max_obs, 2), dtype=torch.float32)
    obs_y_norm_tensor      = torch.zeros((N, max_obs), dtype=torch.float32)

    #mask_tensor      = torch.zeros((N, max_obs), dtype=torch.bool)
    q_coords_tensor  = torch.zeros((N, 2), dtype=torch.float32)
    q_y_tensor       = torch.zeros((N,), dtype=torch.float32)
    q_y_norm_tensor  = torch.zeros((N,), dtype=torch.float32)

    ###########################################################################
    # 3) PREPARE query coordinates
    ###########################################################################
    q_coords = dataset.coords.cpu().double().numpy()
    q_lat = q_coords[:, 0]
    q_lon = q_coords[:, 1]

    q_xy_all = np.stack([q_lon * k_lon, q_lat * k_lat], axis=1)

    ###########################################################################
    # 4) BATCHED radius-query
    ###########################################################################
    batch_size = 10000
    num_batches = (N + batch_size - 1) // batch_size

    print("Starting batched KDTree radius queries...")

    for b in range(num_batches):
        start = b * batch_size
        end   = min(start + batch_size, N)

        batch_xy = q_xy_all[start:end]

        # Perform radius query for this batch
        inds_batch, dists_batch = kdt.query_radius(
            batch_xy,
            r=max_radius_km,
            return_distance=True
        )

        #######################################################################
        # Process each point in batch
        #######################################################################
        for j, (inds, dists) in enumerate(zip(inds_batch, dists_batch)):
            i = start + j  # real index in dataset

            q_coord = dataset.coords[i]
            q_y     = dataset.y[i]

            # Store query
            q_coords_tensor[i] = q_coord
            q_y_tensor[i]      = q_y
            q_y_norm_tensor[i] = (q_y - train_y_mean) / train_y_std

            # FAST self-exclude (if needed)
            if self_exclude:
                mask_dx = dists > 1e-6
                inds  = inds[mask_dx]
                dists = dists[mask_dx]

            # Cap to max_obs nearest neighbors
            # if len(inds) > max_obs:
            #     order = np.argsort(dists)[:max_obs]
            #     inds  = inds[order]
            #     dists = dists[order]
                    # Random
            if len(inds) > 10:
                inds = np.random.choice(inds, size=10, replace=False)

            # Convert to tensor
            if len(inds) > 0:
                tinds = torch.as_tensor(inds, dtype=torch.long)
                obs_coords = nei_coords[tinds]
                obs_y      = nei_y[tinds]

                # FIX: squeeze (num,1) -> (num,)
                if obs_y.ndim == 2 and obs_y.shape[1] == 1:
                    obs_y = obs_y.squeeze(-1)

                num = len(inds)

                # Store padded neighbors
                obs_coords_tensor[i, :num] = obs_coords
                obs_y_tensor[i, :num]      = obs_y

                #mask_tensor[i, :num] = True

                # Normalized versions
                obs_coords_norm_tensor[i, :num] = obs_coords - q_coord
                obs_y_norm_tensor[i, :num] = (obs_y - train_y_mean) / train_y_std

        print(f"Processed batch {b+1}/{num_batches}  ({end}/{N})")

    ###########################################################################
    # 5) ATTACH to dataset (replacing old list-based fields)
    ###########################################################################
    dataset.obs_coords      = obs_coords_tensor
    dataset.obs_y           = obs_y_tensor
    dataset.obs_coords_norm = obs_coords_norm_tensor
    dataset.obs_y_norm      = obs_y_norm_tensor
    #dataset.obs_mask        = mask_tensor

    dataset.query_coords    = q_coords_tensor
    dataset.query_y         = q_y_tensor
    dataset.q_y_norm        = q_y_norm_tensor

    return dataset
def _to1d(x):
    if x is None: return np.array([])
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    x = np.asarray(x).reshape(-1)
    x = x[np.isfinite(x)]  # drop NaN/inf
    return x

def save_y_series(sets_y: dict, csv_path: str):
    """sets_y keys like: y_train, y_train_norm, y_val, y_val_norm, ..."""
    rows = []
    for k, arr in sets_y.items():
        vals = _to1d(arr)
        if vals.size == 0: 
            continue
        # parse key into split + norm flag
        # expects keys like: y_train, y_train_norm
        parts = k.split("_")
        split = parts[1] if len(parts) >= 2 else "unknown"
        is_norm = (len(parts) >= 3 and parts[2] == "norm")
        for v in vals:
            rows.append({"split": split, "is_norm": int(is_norm), "y": float(v)})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[y] saved {len(df)} rows to {csv_path}")
# =====================================
# Main loader with 4-way split (train/valid/test/calib)
# =====================================


def parse_keep_n_dict(s: str) -> dict[str, float]:
    out = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad keep_n_dict entry (missing ':'): {part}")
        k, v = part.split(":", 1)
        k = k.strip().lower()
        try:
            v = float(v.strip())
        except ValueError:
            raise ValueError(f"Bad float for keep_n_dict[{k!r}]: {v!r}")
        if not (0.0 < v <= 1.0):
            raise ValueError(f"keep_n value for {k!r} must be in (0,1], got {v}")
        out[k] = v
    return out

def set_creation_func(dataset, selected_idx, max_radius_km, trainset=None, neighbors_train_only=True):
    set = SpatialDataset(
        coords=dataset.coords[selected_idx].numpy(),
        features=dataset.features[selected_idx].numpy(),
        y=dataset.y[selected_idx].numpy()
    )
    if trainset == None:
        # labels stats (scalars)
        train_y_mean = torch.as_tensor(set.y.mean(), dtype=torch.float32)
        train_y_std  = torch.as_tensor(set.y.std(),  dtype=torch.float32).clamp_min(1e-6)

        set.y_mean = train_y_mean
        set.y_std  = train_y_std
        train_lat_std = set.coords[:,0].std().float().clamp_min(1e-6)
        train_lon_std = set.coords[:,1].std().float().clamp_min(1e-6)
        set.lat_std = train_lat_std
        set.lon_std = train_lon_std

        faster_add_transformer_masks(set, set.coords, set.y, set.y_mean, set.y_std,
                       max_radius_km=max_radius_km, self_exclude=True, max_obs=10, min_k=8)
    else:
        if(neighbors_train_only == False):
             # --- optionally limit to a percentage of extra points ---
            use_ratio = getattr(trainset, "neighbor_ratio", 1.0)  # default 100%
            if use_ratio < 1.0:
                n_extra = int(len(set.coords) * use_ratio)
                sel_extra = torch.randperm(len(set.coords))[:n_extra]
                coords = torch.concat((trainset.coords, set.coords[sel_extra]), dim=0)
                y = torch.concat((trainset.y, set.y[sel_extra]), dim=0)
            else:
                coords = torch.concat((trainset.coords, set.coords), dim=0)
                y = torch.concat((trainset.y, set.y), dim=0)
            faster_add_transformer_masks(set, coords, y, trainset.y_mean, trainset.y_std,
                          max_radius_km=max_radius_km, self_exclude=True, max_obs=10, min_k=8)
        else:
            faster_add_transformer_masks(set, trainset.coords, trainset.y, trainset.y_mean, trainset.y_std,
                        max_radius_km=max_radius_km, self_exclude=True, max_obs=10, min_k=8)
        

    return set


def load_multi_dt2_data(args):
    """
    Load data for training, validation, test, and calibration from a DTED file.

    Returns
    -------
    trainset, validset, testset, calibset : SpatialDataset objects
    """
    # 1. Setup File Paths and Cache Key
    t0 = time.perf_counter()  # start as early as possible
    os.makedirs("Transformer_Map_Interp/cache/", exist_ok=True)
    # The cache key now depends on the names of all input files, not just one base file.
    file_list = [args.train_file, args.valid_file, args.test_file]#, args.calib_file]
    cache_base = "_".join([os.path.basename(f).split('.')[0] for f in file_list])
    cache_key = f"{cache_base}_keep_n{args.keep_n}_seed{args.random_seed}"

    # Define paths for loading DT2Dataset and creating cache keys
    set_configs = {
        "train": os.path.join(args.data_path, args.train_file),
        "valid": os.path.join(args.data_path, args.valid_file),
        "test": os.path.join(args.data_path, args.test_file),
        #"calib": os.path.join(args.data_path, args.calib_file),
    }

    # Verify all input files exist
    for name, path in set_configs.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File for {name} set does not exist: {path}")

    # Check for cache existence
    # cache_exists = all(
    #     os.path.exists(f"Transformer_Map_Interp/cache/{name}set_{cache_key}.pt")
    #     for name in set_configs
    # )
    cache_exists = all(
        os.path.exists(f"Transformer_Map_Interp/cache/{name}set_2_M_points_10_nei_new_saving_with_batching.pt")
        for name in set_configs
    )

    if cache_exists and (args.new_spread == False):
        print("Loading cached sets...")
        # trainset = torch.load(f"Transformer_Map_Interp/cache/trainset_{cache_key}.pt", weights_only=False)
        # validset = torch.load(f"Transformer_Map_Interp/cache/validset_{cache_key}.pt", weights_only=False)
        # testset  = torch.load(f"Transformer_Map_Interp/cache/testset_{cache_key}.pt",  weights_only=False)
        # calibset = torch.load(f"Transformer_Map_Interp/cache/calibset_{cache_key}.pt", weights_only=False)
        trainset = torch.load(f"Transformer_Map_Interp/cache/trainset_2_M_points_10_nei_new_saving_with_batching.pt", weights_only=False)
        validset = torch.load(f"Transformer_Map_Interp/cache/validset_2_M_points_10_nei_new_saving_with_batching.pt", weights_only=False)
        testset  = torch.load(f"Transformer_Map_Interp/cache/testset_2_M_points_10_nei_new_saving_with_batching.pt",  weights_only=False)
        return trainset, validset, testset #, calibset

    print("Creating and caching sets...")
    
    # 2. Load Datasets Individually
    raw_datasets = {}
    for name, dt2_file in set_configs.items():
        print(f"[DEBUG] Loading {name} from: {dt2_file}")
        raw_datasets[name] = DT2Dataset(
            dt2_file=dt2_file, 
            include_elevation_in_features=False, 
            normalize=getattr(args, 'normalize_elev', False)
        )
    # 3. Subsample and Create Final Sets (Deterministic)
    rng = np.random.RandomState(seed=args.random_seed)
    final_sets = {}
    keep_n_dict = parse_keep_n_dict(args.keep_n_dict)

    for name, dataset in raw_datasets.items():
        total = dataset.coords.shape[0]
        # Get the specific keep_n ratio for this set
        keep_ratio = keep_n_dict[name]
        keep_n = int(total * keep_ratio)
        
        
        # Subsample indices deterministically
        selected_idx = rng.choice(total, size=keep_n, replace=False)
        # Create the final SpatialDataset using the selected indices
        # We assume sets_creation_func can handle a single dataset/index list
        if name == 'train':
            print("Creating train set...")
            final_sets[name] = set_creation_func(dataset, selected_idx, args.max_km)
        else:
            print(f"Creating {name} set...")
            final_sets[name] = set_creation_func(dataset, selected_idx, args.max_km, final_sets['train'], neighbors_train_only=args.neighbors_train_only)
        print(f"Num {name}: {total} total, {keep_n} kept ({args.keep_n*100:.1f}%)")

    trainset = final_sets['train']
    validset = final_sets['valid']
    testset = final_sets['test']
    #calibset = final_sets['calib']

    # ------- Inspect & Cache -------
    inspect_dataset(trainset, name="Train")
    inspect_dataset(validset, name="valid")
    inspect_dataset(testset, name="Test")
    
    # sets_y = {
    #     "y_train": trainset.y,
    #     "y_train_norm": trainset.y_norm,
    #     "y_val": validset.y,
    #     "y_val_norm": validset.y_norm,
    #     "y_test": testset.y,
    #     "y_test_norm": testset.y_norm,
    # }
    # save_y_series(sets_y, "y_values.csv")

    # Cache the final SpatialDataset objects
    print("Saving sets...")
    # torch.save(trainset, f"Transformer_Map_Interp/cache/trainset_2_M_points_10_nei_new_saving_with_batching.pt")
    # torch.save(validset, f"Transformer_Map_Interp/cache/validset_2_M_points_10_nei_new_saving_with_batching.pt")
    # torch.save(testset,  f"Transformer_Map_Interp/cache/testset_2_M_points_10_nei_new_saving_with_batching.pt")
    torch.save(trainset, f"Transformer_Map_Interp/cache/trainset_blobed.pt")
    torch.save(validset, f"Transformer_Map_Interp/cache/validset_blobed.pt")
    torch.save(testset,  f"Transformer_Map_Interp/cache/testset_blobed.pt")
    #torch.save(calibset, f"Transformer_Map_Interp/cache/calibset_{cache_key}.pt")
    print (f"build Kdtree Lap: {time.perf_counter() - t0:.3f}s")
    return trainset, validset, testset#, calibset