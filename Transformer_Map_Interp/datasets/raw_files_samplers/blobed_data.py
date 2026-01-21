# Transformer_Map_Interp/datasets/make_dt2_splits.py
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import rasterio
print(os.getcwd())
sys.path.append(os.path.abspath(""))  # project root
# os.chdir("../../")
# print(os.getcwd())
from Transformer_Map_Interp.datasets.dt2_data import DT2Dataset

# ----------------------------
# CONFIG
# ----------------------------
DT2_PATH   = "./Transformer_Map_Interp/datasets/merged.tif"

OUT_DIR    = "./Transformer_Map_Interp/datasets"   # where to put train/val/test tiffs
TRAIN_TIF  = os.path.join(OUT_DIR, "train_blobs_new.tiff")
VAL_TIF    = os.path.join(OUT_DIR, "val_blobs_new.tiff")
TEST_TIF   = os.path.join(OUT_DIR, "test_blobs_new.tiff")

FIG_DIR    = "./Transformer_Map_Interp/figures"

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEA_THRESHOLD_M = 0.0      # treat y <= 5 m as "sea"
N_CLUSTERS = 80            # number of spatial blobs
SEED       = 5

# ----------------------------
# 1. Load DT2 + remove sea
# ----------------------------
print(f"Loading DT2 from {DT2_PATH}")
dt2 = DT2Dataset(DT2_PATH, include_elevation_in_features=False)

coords   = dt2.coords.numpy()          # (N, 2) [lat, lon]
y        = dt2.y.numpy().reshape(-1)   # (N,)

# land only (for splitting)
land_mask = y > SEA_THRESHOLD_M
coords_land = coords[land_mask]
y_land      = y[land_mask]

print(f"Total points: {len(y)}, land points: {len(y_land)}")

# We also need the original 2D raster & metadata to write new tiffs
with rasterio.open(DT2_PATH) as src:
    elevation = src.read(1)           # (H, W)
    profile   = src.profile
    height, width = elevation.shape

# flatten elevation to align with y
elev_flat = elevation.flatten()
assert elev_flat.shape[0] == y.shape[0]

land_mask_flat = land_mask           # alias; already 1D over all pixels
land_indices   = np.where(land_mask_flat)[0]  # positions in the full raster

# ----------------------------
# 2. KMeans spatial blobs (on land coords)
# ----------------------------
print("Running KMeans for spatial blobs...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
cluster_ids = kmeans.fit_predict(coords_land)   # (N_land,)

# ----------------------------
# 3. Assign clusters to splits (round-robin by mean elevation)
# ----------------------------
train_ratio, val_ratio, test_ratio = TRAIN_RATIO, VAL_RATIO, TEST_RATIO
assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

cluster_means = []
for cid in range(N_CLUSTERS):
    mask = (cluster_ids == cid)
    if not np.any(mask):
        continue
    cluster_means.append((cid, float(y_land[mask].mean())))

# sort clusters by elevation
cluster_means.sort(key=lambda x: x[1])

train_clusters, val_clusters, test_clusters = [], [], []

# --- VVVVVV MODIFICATION STARTS HERE VVVVVV ---
# 2. Assign using a repeating pattern of 4: [Train, Train, Val, Test]
# This ensures 50% Train, 25% Val, 25% Test, evenly spread across elevations.
for i, (cid, _) in enumerate(cluster_means):
    pattern_idx = i % 4
    
    if pattern_idx == 1:
        # Indices 0 and 1 go to Train (2 out of 4 = 50%)
        val_clusters.append(cid)
    elif pattern_idx == 0:
        # Index 2 goes to Val (1 out of 4 = 25%)
        test_clusters.append(cid)
    else:
        # Index 3 goes to Test (1 out of 4 = 25%)
        train_clusters.append(cid)

cluster_ids = np.asarray(cluster_ids)
train_mask_land = np.isin(cluster_ids, train_clusters)
val_mask_land   = np.isin(cluster_ids, val_clusters)
test_mask_land  = np.isin(cluster_ids, test_clusters)
# # Target cluster percentages
# target_train_pct = 0.50
# target_val_pct = 0.25
# # target_test_pct = 0.25

# # Get the list of cluster IDs sorted by mean elevation
# sorted_cluster_ids = [cid for cid, _ in cluster_means]

# # We iterate through the sorted clusters and assign them based on their rank (index i)
# # to achieve the 50/25/25 distribution while keeping the elevation balanced.
# for i, cid in enumerate(sorted_cluster_ids):
#     # Calculate the cluster's rank as a percentage of the total number of clusters
#     # i.e., what percentage of clusters are at or below this one's mean elevation
#     rank_pct = (i + 1) / len(sorted_cluster_ids)
    
#     # Assign the cluster based on its rank percentage
#     if rank_pct <= target_train_pct:
#         # Assign first 50% of clusters (by elevation) to the Train set
#         train_clusters.append(cid)
#     elif rank_pct <= (target_train_pct + target_val_pct):
#         # Assign next 25% of clusters to the Val set (between 50% and 75% rank)
#         val_clusters.append(cid)
#     else:
#         # Assign remaining 25% to the Test set (above 75% rank)
#         test_clusters.append(cid)

# --- ^^^^^^ MODIFICATION ENDS HERE ^^^^^^ ---

# for i, (cid, _) in enumerate(cluster_means):
#     r = i % 3
#     if r == 0:
#         train_clusters.append(cid)
#     elif r == 1:
#         val_clusters.append(cid)
#     else:
#         test_clusters.append(cid)

cluster_ids = np.asarray(cluster_ids)
train_mask_land = np.isin(cluster_ids, train_clusters)
val_mask_land   = np.isin(cluster_ids, val_clusters)
test_mask_land  = np.isin(cluster_ids, test_clusters)

train_idx_land = np.where(train_mask_land)[0]
val_idx_land   = np.where(val_mask_land)[0]
test_idx_land  = np.where(test_mask_land)[0]

print(f"Train land points: {len(train_idx_land)}")
print(f"Val   land points: {len(val_idx_land)}")
print(f"Test  land points: {len(test_idx_land)}")

# Map from land indices back to full-raster flat indices
train_global_idx = land_indices[train_idx_land]
val_global_idx   = land_indices[val_idx_land]
test_global_idx  = land_indices[test_idx_land]

# ----------------------------
# 4. Plots: spatial and histograms (sanity checks)
# ----------------------------
os.makedirs(FIG_DIR, exist_ok=True)

# Spatial split (scatter of land points)
plt.figure(figsize=(6, 8))
plt.scatter(coords_land[train_idx_land, 1], coords_land[train_idx_land, 0],
            s=2, c="tab:blue", label="Train", alpha=0.6)
plt.scatter(coords_land[val_idx_land, 1],   coords_land[val_idx_land, 0],
            s=4, c="tab:orange", label="Val", alpha=0.8)
plt.scatter(coords_land[test_idx_land, 1],  coords_land[test_idx_land, 0],
            s=4, c="tab:green", label="Test", alpha=0.8)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Split (land only)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "dt2_spatial_split_blobs_new.png"), dpi=200)
plt.close()

# Elevation histograms per split
bins = 40
plt.figure(figsize=(8, 6))
plt.hist(y_land[train_idx_land], bins=bins, density=True, alpha=0.4, label="Train")
plt.hist(y_land[val_idx_land],   bins=bins, density=True, alpha=0.4, label="Val")
plt.hist(y_land[test_idx_land],  bins=bins, density=True, alpha=0.4, label="Test")
plt.xlabel("Elevation (m)")
plt.ylabel("Density")
plt.legend()
plt.title("Elevation Distributions per Split (land only)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "dt2_elevation_hist_per_split_blobs_new.png"), dpi=200)
plt.close()

print(f"Saved figures in {FIG_DIR}/")

# ----------------------------
# 5. Build 3 raster masks & save as GeoTIFFs
# ----------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# Use a nodata value distinct from valid elevations
nodata_val = profile.get("nodata")
if nodata_val is None:
    nodata_val = -9999
profile.update(count=1, nodata=nodata_val)

# start with all nodata
train_arr = np.full_like(elevation, nodata_val)
val_arr   = np.full_like(elevation, nodata_val)
test_arr  = np.full_like(elevation, nodata_val)

# compute row/col for each global flat index
train_rows, train_cols = np.divmod(train_global_idx, width)
val_rows,   val_cols   = np.divmod(val_global_idx,   width)
test_rows,  test_cols  = np.divmod(test_global_idx,  width)

# copy original elevation into the appropriate split
train_arr[train_rows, train_cols] = elevation[train_rows, train_cols]
val_arr[val_rows,     val_cols]   = elevation[val_rows,   val_cols]
test_arr[test_rows,   test_cols]  = elevation[test_rows,  test_cols]

# sea pixels stay nodata in all three

# Write each split as a GeoTIFF
for path, arr, name in [
    (TRAIN_TIF, train_arr, "train"),
    (VAL_TIF,   val_arr,   "val"),
    (TEST_TIF,  test_arr,  "test"),
]:
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    print(f"Wrote {name} split GeoTIFF to: {path}")

print("Done.")
