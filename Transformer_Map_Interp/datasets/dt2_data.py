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
    # add_transformer_masks(trainset, trainset.coords, trainset.y, trainset.y_mean, trainset.y_std,
    #                   max_radius_km=max_radius_km, self_exclude=True, max_obs=256, min_k=8)
    # add_transformer_masks(validset, trainset.coords, trainset.y, trainset.y_mean, trainset.y_std,
    #                     max_radius_km=max_radius_km, self_exclude=False,max_obs=256, min_k=8)
    # add_transformer_masks(testset, trainset.coords, trainset.y, trainset.y_mean, trainset.y_std,
    #                     max_radius_km=max_radius_km, self_exclude=False,max_obs=256, min_k=8)
    # add_transformer_masks(calibset, trainset.coords, trainset.y, trainset.y_mean, trainset.y_std,
    #                     max_radius_km=max_radius_km, self_exclude=False,max_obs=256, min_k=8)
    faster_add_transformer_masks(trainset, trainset.coords, trainset.y, trainset.y_mean, trainset.y_std,
                       max_radius_km=max_radius_km, self_exclude=True, max_obs=256, min_k=8)
    faster_add_transformer_masks(validset, trainset.coords, trainset.y, trainset.y_mean, trainset.y_std,
                         max_radius_km=max_radius_km, self_exclude=False,max_obs=256, min_k=8)
    faster_add_transformer_masks(testset, trainset.coords, trainset.y, trainset.y_mean, trainset.y_std,
                         max_radius_km=max_radius_km, self_exclude=False,max_obs=256, min_k=8)
    faster_add_transformer_masks(calibset, trainset.coords, trainset.y, trainset.y_mean, trainset.y_std,
                         max_radius_km=max_radius_km, self_exclude=False,max_obs=256, min_k=8)
    return trainset, validset, testset, calibset

def add_transformer_masks(
    dataset,
    train_coords,
    train_y,
    train_y_mean, 
    train_y_std,
    max_radius_km=None,
    self_exclude=False,
    max_obs=None,         # cap number of observed tokens (optional)
    min_k=8,              # fallback K if radius leaves too few neighbors
    coord_atol=1e-9       # tolerance for coordinate equality
):
    # helpers
    def haversine_dist(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = torch.deg2rad(lat2 - lat1)
        dlon = torch.deg2rad(lon2 - lon1)
        a = torch.sin(dlat/2)**2 + torch.cos(torch.deg2rad(lat1)) * torch.cos(torch.deg2rad(lat2)) * torch.sin(dlon/2)**2
        return R * (2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a)))

    dataset.obs_coords = []
    dataset.obs_y = []
    dataset.query_coords = []
    dataset.query_y = []
    dataset.obs_y_norm = []
    dataset.obs_coords_norm = []
    dataset.q_y_norm = []
    #dataset.obs_mask = []
    #dataset.query_mask = []

    # ensure tensors
    train_coords = train_coords.clone()
    train_y = train_y.clone()
    dataset.y_norm = (dataset.y - train_y_mean) / train_y_std
    
    for i in range(len(dataset)):
        q_coord = dataset.coords[i]
        q_y = dataset.y[i]

        # start with all train points
        if max_radius_km is not None:
            dists = haversine_dist(q_coord[0], q_coord[1], train_coords[:, 0], train_coords[:, 1])
            keep = dists <= max_radius_km
        else:
            keep = torch.ones(train_coords.shape[0], dtype=torch.bool)

        # exclude the query point itself (only for train set)
        if self_exclude:
            same_lat = torch.isclose(train_coords[:, 0], q_coord[0], atol=coord_atol)
            same_lon = torch.isclose(train_coords[:, 1], q_coord[1], atol=coord_atol)
            keep = keep & ~(same_lat & same_lon)

        # if nothing (or too few) remains, fall back to nearest min_k neighbors
        if keep.sum().item() < min_k:
            # compute distances once (if we didn't already)
            if max_radius_km is None:
                dists = haversine_dist(q_coord[0], q_coord[1], train_coords[:, 0], train_coords[:, 1])
            # exclude self from fallback set as well
            if self_exclude:
                dists[same_lat & same_lon] = float('inf')
            k = min(min_k, (train_coords.shape[0] - (1 if self_exclude else 0)))
            topk = torch.topk(-dists, k).indices  # negative -> smallest distance
            keep = torch.zeros_like(keep); keep[topk] = True

        obs_coords = train_coords[keep]
        obs_y = train_y[keep]

        # optionally cap max observed tokens (helps memory)
        if max_obs is not None and obs_coords.shape[0] > max_obs:
            # choose closest max_obs
            dists = haversine_dist(q_coord[0], q_coord[1], obs_coords[:, 0], obs_coords[:, 1])
            sel = torch.topk(-dists, max_obs).indices
            obs_coords = obs_coords[sel]
            obs_y = obs_y[sel]

        # build masks
        #L = obs_coords.shape[0] + 1  # +1 for CLS token
        #obs_mask = torch.zeros(L, dtype=torch.bool); obs_mask[:L-1] = True
        #query_mask = torch.zeros(L, dtype=torch.bool); query_mask[-1] = True

        # Normalized
        obs_coords_norm = obs_coords - q_coord
        obs_y_norm = (obs_y - train_y_mean) / train_y_std
        q_y_norm = (q_y - train_y_mean) / train_y_std

        # --- enforce consistent shapes ---
        # neighbors: (S,) not (S,1)
        if obs_y.ndim == 2 and obs_y.size(-1) == 1:
            obs_y = obs_y.squeeze(-1)
        if obs_y_norm.ndim == 2 and obs_y_norm.size(-1) == 1:
            obs_y_norm = obs_y_norm.squeeze(-1)

        # query y: scalar ()
        q_y = q_y.squeeze()
        q_y_norm = q_y_norm.squeeze()
        
        dataset.obs_coords.append(obs_coords)
        dataset.obs_y.append(obs_y)
        dataset.query_coords.append(q_coord)
        dataset.query_y.append(q_y)
        dataset.obs_coords_norm.append(obs_coords_norm)
        dataset.obs_y_norm.append(obs_y_norm)
        dataset.q_y_norm.append(q_y_norm)

    # --- finalize per-target tensors (uniform length N) ---
    if isinstance(dataset.query_coords, list):
        dataset.query_coords = torch.stack(
            [torch.as_tensor(x, dtype=torch.float32) for x in dataset.query_coords], dim=0
        )  # (N, 2)
        
    # if isinstance(dataset.obs_coords_norm, list):
    #     dataset.obs_coords_norm = torch.stack(
    #         [torch.as_tensor(x, dtype=torch.float32) for x in dataset.query_coords], dim=0
    #     )  # (N, 2)
    
    # if isinstance(dataset.obs_coords, list):
    #     dataset.obs_coords = torch.stack(
    #         [torch.as_tensor(x, dtype=torch.float32) for x in dataset.query_coords], dim=0
    #     )  # (N, 2)

    if isinstance(dataset.q_y_norm, list):
        dataset.q_y_norm = torch.stack(
            [torch.as_tensor(x, dtype=torch.float32).reshape(1) for x in dataset.q_y_norm], dim=0
        ).squeeze(-1)  # (N,)
        
    if isinstance(dataset.query_y, list):
        dataset.query_y = torch.stack(
            [torch.as_tensor(x, dtype=torch.float32).reshape(1) for x in dataset.q_y], dim=0
        ).squeeze(-1)
        
    # if isinstance(dataset.obs_y_norm, list):
    #     dataset.obs_y_norm = torch.stack(
    #         [torch.as_tensor(x, dtype=torch.float32).reshape(1) for x in dataset.q_y_norm], dim=0
    #     ).squeeze(-1)
            
    # if isinstance(dataset.obs_y, list):
    #     dataset.obs_y = torch.stack(
    #         [torch.as_tensor(x, dtype=torch.float32).reshape(1) for x in dataset.q_y_norm], dim=0
    #     ).squeeze(-1)   
    
    
        #dataset.obs_mask.append(obs_mask[:L-1])  # keep mask per observed only if you prefer
        #dataset.query_mask.append(query_mask)
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
    # ---- 0) Build a single KDTree on TRAIN coords in a planar (km) space ----
    # Equirectangular projection with fixed reference latitude:
    #   x_km = (lon_deg) * (111.320 * cos(lat0_deg))
    #   y_km = (lat_deg) * 110.574
    with torch.no_grad():
        nei_lat_deg = nei_coords[:, 0].cpu().double().numpy()
        nei_lon_deg = nei_coords[:, 1].cpu().double().numpy()
    lat0 = float(np.mean(nei_lat_deg))
    k_lat = 110.574                # km per 1 degree latitude
    k_lon = 111.320 * np.cos(np.deg2rad(lat0))  # km per 1 degree longitude at lat0

    nei_xy_km = np.stack([nei_lon_deg * k_lon, nei_lat_deg * k_lat], axis=1)
    # if(neighbors_train_only == True and set=='train'):
    print("Building KDTree on neighbor coords...")
    kdt = KDTree(nei_xy_km, metric='euclidean')
       
    # else:
    #     //take using percentage_from_target points out of the target set
    #     // concat with the trainging data
    #     kdt = KDTree(train_plus_target_xy_km, metric='euclidean')
        
    dataset.obs_coords = []
    dataset.obs_y = []
    dataset.query_coords = []
    dataset.query_y = []
    dataset.obs_y_norm = []
    dataset.obs_coords_norm = []
    dataset.q_y_norm = []
    #dataset.obs_mask = []
    #dataset.query_mask = []

    # ensure tensors
    #train_coords = train_coords.clone()
    #train_y = train_y.clone()
    dataset.y_norm = (dataset.y - train_y_mean) / train_y_std
    N_nei = nei_coords.shape[0]

    # The NEWER, FASTER APPROACH: single KDTree query per target point
    # Inside faster_add_transformer_masks, before the loop
    q_coords_all = dataset.coords.cpu().double().numpy()
    q_lat_all = q_coords_all[:, 0]
    q_lon_all = q_coords_all[:, 1]

    # This is your k_lat and k_lon from the train_coords
    q_xy_all = np.stack([q_lon_all * k_lon, q_lat_all * k_lat], axis=1)
    
    # Query all points at once for radius
    if max_radius_km is not None:
        # This returns a list of arrays, one array of indices per query point
        print("Starting batch radius query...")
        all_inds_list = kdt.query_radius(q_xy_all, r=max_radius_km, return_distance=False)
        print(f"Number of neighbors per query (first 10): {[len(inds) for inds in all_inds_list[:10]]}")
    else:
        # Fallback: just return indices for all N_train points for each query
        all_inds_list = [np.arange(N_nei) for _ in range(len(q_xy_all))]

    # Run a separate batch query for the min_k fallback (if needed)
    # This returns one big array (N_queries, k)
    # all_inds_knn = kdt.query(q_xy_all, k=min(min_k + 1, Ntrain), return_distance=False)
    
    # Now loop through the *results*
    for i in range(len(dataset)):
        q_coord = dataset.coords[i]
        q_y = dataset.y[i]

        inds = all_inds_list[i] # Get the pre-computed indices

        # --- self-exclude logic (can also be vectorized) ---
        if self_exclude:
            same_lat = torch.isclose(nei_coords[:, 0], q_coord[0], atol=coord_atol).cpu().numpy()
            same_lon = torch.isclose(nei_coords[:, 1], q_coord[1], atol=coord_atol).cpu().numpy()
            same_pt_mask = same_lat & same_lon
            if inds.size == N_nei:
                inds = np.where(~same_pt_mask)[0]
            else:
                inds = inds[~same_pt_mask[inds]]

        # # --- ensure at least min_k via kNN fallback ---
        # if (inds is None) or (len(inds) < min_k):
        #     ind_knn = all_inds_knn[i]
        #     # ... (your logic to process ind_knn) ...
        #     inds = ind_knn[:min_k] 

        # --- slice neighbor data ---
        torch_inds = torch.as_tensor(inds, dtype=torch.long)
        obs_coords = nei_coords[torch_inds]
        obs_y = nei_y[torch_inds]

    # ... (rest of your normalization and appending logic) ...
    
    # The unoptimized way: loop over target points
    # for i in range(len(dataset)):
    #     q_coord = dataset.coords[i]
    #     q_y = dataset.y[i]

    #     q_lat = float(q_coord[0].item())
    #     q_lon = float(q_coord[1].item())
    #     q_xy  = np.array([[q_lon * k_lon, q_lat * k_lat]], dtype=float)
    #     ## --- 1) radius neighbors if specified ---
    #     if max_radius_km is not None:
    #         inds = kdt.query_radius(q_xy, r=max_radius_km, return_distance=False)[0]
    #         if (i%1000 == 0):
    #             print(f"Number of neighbors:{len(inds)}")
    #     else:
    #         # consider all points if no radius constraint
    #         inds = np.arange(Ntrain)

        # # --- self-exclude (exact coord match in degrees) ---
        # if self_exclude:
        #     same_lat = torch.isclose(train_coords[:, 0], q_coord[0], atol=coord_atol).cpu().numpy()
        #     same_lon = torch.isclose(train_coords[:, 1], q_coord[1], atol=coord_atol).cpu().numpy()
        #     same_pt_mask = same_lat & same_lon
        #     if inds.size == Ntrain:
        #         inds = np.where(~same_pt_mask)[0]
        #     else:
        #         inds = inds[~same_pt_mask[inds]]

        # # --- 2) ensure at least min_k via kNN fallback ---
        # if (inds is None) or (len(inds) < min_k):
        #     print("second query")
        #     k = min(min_k + (1 if self_exclude else 0), Ntrain)
        #     # distances ignored; we only need indices
        #     ind_knn = kdt.query(q_xy, k=k, return_distance=False)[0]
        #     if self_exclude:
        #         same_lat = torch.isclose(train_coords[ind_knn, 0], q_coord[0], atol=coord_atol).cpu().numpy()
        #         same_lon = torch.isclose(train_coords[ind_knn, 1], q_coord[1], atol=coord_atol).cpu().numpy()
        #         mask = ~(same_lat & same_lon)
        #         ind_knn = ind_knn[mask]
        #     inds = ind_knn[:min_k] if len(ind_knn) > min_k else ind_knn

        # --- slice train data ---
        # torch_inds = torch.as_tensor(inds, dtype=torch.long)
        # obs_coords = train_coords[torch_inds]
        # obs_y      = train_y[torch_inds]

        # --- optionally cap to max_obs by closest neighbors (kNN over the current set) ---
        # if (max_obs is not None) and (obs_coords.shape[0] > max_obs):
        #     # Get distances for these neighbors and keep the nearest max_obs
        #     # Fast way: query k = len(inds) and match; or compute directly in km-space
        #     sub_xy = np.stack(
        #         [obs_coords[:, 1].cpu().double().numpy() * k_lon,
        #          obs_coords[:, 0].cpu().double().numpy() * k_lat],
        #         axis=1
        #     )
        #     d2 = np.sum((sub_xy - q_xy[0])**2, axis=1)  # squared euclidean
        #     order = np.argsort(d2)[:max_obs]
        #     torch_inds = torch_inds[torch.as_tensor(order, dtype=torch.long)]
        #     obs_coords = train_coords[torch_inds]
        #     obs_y      = train_y[torch_inds]

        # Normalized
        obs_coords_norm = obs_coords - q_coord
        obs_y_norm = (obs_y - train_y_mean) / train_y_std
        q_y_norm = (q_y - train_y_mean) / train_y_std

        # --- enforce consistent shapes ---
        # ensure shapes (keep your existing squeeze logic)
        if obs_y.ndim == 2 and obs_y.size(-1) == 1:
            obs_y = obs_y.squeeze(-1)
        if obs_y_norm.ndim == 2 and obs_y_norm.size(-1) == 1:
            obs_y_norm = obs_y_norm.squeeze(-1)
        q_y = q_y.squeeze()
        q_y_norm = q_y_norm.squeeze()

        # append to your ragged lists
        dataset.obs_coords.append(obs_coords)
        dataset.obs_y.append(obs_y)
        dataset.query_coords.append(q_coord)
        dataset.query_y.append(q_y)
        dataset.obs_coords_norm.append(obs_coords_norm)
        dataset.obs_y_norm.append(obs_y_norm)
        dataset.q_y_norm.append(q_y_norm)

    # --- finalize per-target tensors (uniform length N) ---
    if isinstance(dataset.query_coords, list):
        dataset.query_coords = torch.stack(
            [torch.as_tensor(x, dtype=torch.float32) for x in dataset.query_coords], dim=0
        )  # (N, 2)
        
    # if isinstance(dataset.obs_coords_norm, list):
    #     dataset.obs_coords_norm = torch.stack(
    #         [torch.as_tensor(x, dtype=torch.float32) for x in dataset.query_coords], dim=0
    #     )  # (N, 2)
    
    # if isinstance(dataset.obs_coords, list):
    #     dataset.obs_coords = torch.stack(
    #         [torch.as_tensor(x, dtype=torch.float32) for x in dataset.query_coords], dim=0
    #     )  # (N, 2)

    if isinstance(dataset.q_y_norm, list):
        dataset.q_y_norm = torch.stack(
            [torch.as_tensor(x, dtype=torch.float32).reshape(1) for x in dataset.q_y_norm], dim=0
        ).squeeze(-1)  # (N,)
        
    if isinstance(dataset.query_y, list):
        dataset.query_y = torch.stack(
            [torch.as_tensor(x, dtype=torch.float32).reshape(1) for x in dataset.q_y_norm], dim=0
        ).squeeze(-1)
        
import os
import numpy as np
import pandas as pd
import torch

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

def load_dt2_data(args):
    """
    Load data for training, validation, test, and calibration from a DTED file.

    Returns
    -------
    trainset, validset, testset, calibset : SpatialDataset objects
    """
    # file path
    os.makedirs("Transformer_Map_Interp/cache/", exist_ok=True)
    cache_key = f"{args.dataset}_keep_n{args.keep_n}"
    dt2_file = os.path.join(args.data_path, args.dataset + ".tiff")
    print(f"[DEBUG] Using dt2_file path: {dt2_file}")
    assert os.path.isfile(dt2_file), f"File does not exist: {dt2_file}"

    cache_exists = (
        os.path.exists(f"Transformer_Map_Interp/cache/trainset_{cache_key}.pt") and
        os.path.exists(f"Transformer_Map_Interp/cache/validset_{cache_key}.pt") and
        os.path.exists(f"Transformer_Map_Interp/cache/testset_{cache_key}.pt") and
        os.path.exists(f"Transformer_Map_Interp/cache/calibset_{cache_key}.pt")
    )

    if cache_exists and (args.new_spread == False):
        print("Loading cached sets...")
        trainset = torch.load(f"Transformer_Map_Interp/cache/trainset_{cache_key}.pt", weights_only=False)
        validset = torch.load(f"Transformer_Map_Interp/cache/validset_{cache_key}.pt", weights_only=False)
        testset  = torch.load(f"Transformer_Map_Interp/cache/testset_{cache_key}.pt",  weights_only=False)
        calibset = torch.load(f"Transformer_Map_Interp/cache/calibset_{cache_key}.pt", weights_only=False)
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
        trainset, validset, testset, calibset = sets_creation_func(dataset, idx_tr, idx_va, idx_te, idx_ca, args.max_km)

    elif (args.datasampling == 'normal' and args.setsdistribtuion == 'equal'):
        sel = selected_ind_normal(dataset, 0, keep_n, args)
        perm = rng.permutation(len(sel))
        idx_tr = sel[perm[:num_train]]
        idx_va = sel[perm[num_train:num_train + num_valid]]
        idx_te = sel[perm[num_train + num_valid:num_train + num_valid + num_test]]
        idx_ca = sel[perm[num_train + num_valid + num_test:]]
        trainset, validset, testset, calibset = sets_creation_func(dataset, idx_tr, idx_va, idx_te, idx_ca, args.max_km)

    elif (args.datasampling == 'normal' and args.setsdistribtuion == 'diff'):
        idx_tr = selected_ind_normal(dataset, mu=0, size=num_train, args=args)
        idx_va = selected_ind_normal(dataset, mu=args.sampling_mu, size=num_valid, args=args, exclude_idx=idx_tr)
        idx_te = selected_ind_normal(dataset, mu=args.sampling_mu, size=num_test, args=args, exclude_idx=np.concatenate([idx_tr, idx_va]))
        idx_ca = selected_ind_normal(dataset, mu=args.sampling_mu, size=num_calib, args=args, exclude_idx=np.concatenate([idx_tr, idx_va, idx_te]))
        trainset, validset, testset, calibset = sets_creation_func(dataset, idx_tr, idx_va, idx_te, idx_ca, args.max_km)

    else:
        raise ValueError("Unsupported combination for datasampling/setsdistribtuion")

    print(f"num_total: {num_total_dataset}, num_train: {num_train}, num_val: {num_valid}, num_test: {num_test}, num_calib: {num_calib}")

    # ------- Normalize targets by train statistics -------
    # y_mean = trainset.y.mean(dim=0, keepdim=True) //Orit: removed - 13.10
    # y_std  = trainset.y.std(dim=0, keepdim=True) + 1e-6

    # trainset.y = (trainset.y - y_mean) / y_std
    # validset.y = (validset.y - y_mean) / y_std
    # testset.y  = (testset.y  - y_mean) / y_std
    # calibset.y = (calibset.y - y_mean) / y_std

    # # TODO: Optional but I think here it is needed
    # trainset.obs_y = (trainset.obs_y - y_mean) / y_std
    # validset.obs_y = (validset.obs_y - y_mean) / y_std
    # testset.obs_y  = (testset.obs_y  - y_mean) / y_std
    # calibset.obs_y = (calibset.obs_y - y_mean) / y_std

    # # Stats for Δcoord standardization
    # self.lat_std = trainset.coords[:, 0].std().item() + 1e-6
    # self.lon_std = trainset.coords[:, 1].std().item() + 1e-6

    # # Relative coords
    # trainset.obs_coords = (mem_coords[:, 0] - cls_coord[0]) / self.lat_std
    # dlon = (mem_coords[:, 1] - cls_coord[1]) / self.lon_std


    # Keep for inverse-transform if needed
    # trainset.y_mean = y_mean //Orit: removed - 13.10
    # trainset.y_std  = y_std

    # ------- Inspect & Cache -------
    inspect_dataset(trainset, name="Train")
    inspect_dataset(testset, name="Test")
    sets_y = {
        "y_train": trainset.y,
        "y_train_norm":   trainset.y_norm,
        "y_val":  validset.y,
        "y_val_norm": validset.y_norm,
        "y_test":  testset.y,
        "y_test_norm": testset.y_norm,
    }
    print(f"Elevation shape is: {trainset.y.shape}\n")
    save_y_series(sets_y, "y_values.csv")
    torch.save(trainset, f"Transformer_Map_Interp/cache/trainset_{cache_key}.pt")
    torch.save(validset, f"Transformer_Map_Interp/cache/validset_{cache_key}.pt")  # (fix) save validset correctly
    torch.save(testset,  f"Transformer_Map_Interp/cache/testset_{cache_key}.pt")
    torch.save(calibset, f"Transformer_Map_Interp/cache/calibset_{cache_key}.pt")

    return trainset, validset, testset, calibset

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
                       max_radius_km=max_radius_km, self_exclude=True, max_obs=256, min_k=8)
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
                          max_radius_km=max_radius_km, self_exclude=True, max_obs=256, min_k=8)
        else:
            faster_add_transformer_masks(set, trainset.coords, trainset.y, trainset.y_mean, trainset.y_std,
                        max_radius_km=max_radius_km, self_exclude=True, max_obs=256, min_k=8)
        

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
    cache_exists = all(
        os.path.exists(f"Transformer_Map_Interp/cache/{name}set_{cache_key}.pt")
        for name in set_configs
    )

    if cache_exists and (args.new_spread == False):
        print("Loading cached sets...")
        trainset = torch.load(f"Transformer_Map_Interp/cache/trainset_{cache_key}.pt", weights_only=False)
        validset = torch.load(f"Transformer_Map_Interp/cache/validset_{cache_key}.pt", weights_only=False)
        testset  = torch.load(f"Transformer_Map_Interp/cache/testset_{cache_key}.pt",  weights_only=False)
        # calibset = torch.load(f"Transformer_Map_Interp/cache/calibset_{cache_key}.pt", weights_only=False)
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
    inspect_dataset(testset, name="Test")
    
    sets_y = {
        "y_train": trainset.y,
        "y_train_norm": trainset.y_norm,
        "y_val": validset.y,
        "y_val_norm": validset.y_norm,
        "y_test": testset.y,
        "y_test_norm": testset.y_norm,
    }
    save_y_series(sets_y, "y_values.csv")

    # Cache the final SpatialDataset objects
    torch.save(trainset, f"Transformer_Map_Interp/cache/trainset_{cache_key}.pt")
    torch.save(validset, f"Transformer_Map_Interp/cache/validset_{cache_key}.pt")
    torch.save(testset,  f"Transformer_Map_Interp/cache/testset_{cache_key}.pt")
    #torch.save(calibset, f"Transformer_Map_Interp/cache/calibset_{cache_key}.pt")
    print (f"build Kdtree Lap: {time.perf_counter() - t0:.3f}s")
    return trainset, validset, testset#, calibset