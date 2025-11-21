import torch
import numpy as np
from pathlib import Path
import torch
import sys, os
print(os.getcwd())
sys.path.append(os.path.abspath("./"))  # project root
print(os.path.abspath("./"))
from Transformer_Map_Interp.datasets.data import SpatialDataset
from Transformer_Map_Interp.datasets.transformerRegressorDataClass import TransformerPointDataset
from scipy.interpolate import Rbf
from scipy.interpolate import LinearNDInterpolator
# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
p_cache = Path("Transformer_Map_Interp/cache")
suffix = "_10neighbors"   # appended to saved files
use_fixed_n = True         # True → keep exactly 10 neighbors, False → keep 10% of them
n_keep = 10
keep_ratio = 0.1
rng = np.random.default_rng(42)   # reproducibility

# ---------------------------------------------------------
# Helper: downsample neighbors
# ---------------------------------------------------------
def downsample_neighbors(dataset, use_fixed_n, n_keep,keep_ratio=0.1):
    """Randomly select subset of neighbors for each sample in the dataset."""
    #obs_coords, obs_y, obs_coords_norm, obs_y_norm=[], [], [], []
    obs_coords, obs_y, obs_coords_norm, obs_y_norm= dataset.obs_coords, dataset.obs_y, dataset.obs_coords_norm,  dataset.obs_y_norm
    n_samples = len(obs_coords)
    new_coords, new_y, new_coords_norm, new_y_norm= [], [], [], []

    for i in range(n_samples):
        n_neighbors = obs_coords[i].shape[0]
        if n_neighbors == 0:
            new_coords.append(obs_coords[i])
            new_y.append(obs_y[i])
            continue

        if use_fixed_n:
            k = min(n_keep, n_neighbors)
        else:
            k = max(1, int(np.ceil(n_neighbors * keep_ratio)))

        chosen_idx = rng.choice(n_neighbors, size=k, replace=False)
        new_coords.append(obs_coords[i][chosen_idx])
        new_y.append(obs_y[i][chosen_idx])
        new_coords_norm.append(obs_coords_norm[i][chosen_idx])
        new_y_norm.append(obs_y_norm[i][chosen_idx])

    # Stack into tensors again
    dataset.obs_coords = new_coords
    dataset.obs_y = new_y
    dataset.obs_coords_norm = new_coords_norm
    dataset.obs_y_norm = new_y_norm
    return dataset

# ---------------------------------------------------------
# Process train/val/test sets
# ---------------------------------------------------------
for split in ["trainset", "validset", "testset"]:
    fname = f"{split}_n32_e035_1arc_v3_cropped_train_n32_e035_1arc_v3_cropped_val_n32_e035_1arc_v3_cropped_test_keep_n0.05_seed5.pt"
    fpath = p_cache / fname
    print(f"Loading {fpath}")
    dataset = torch.load(fpath, map_location="cpu", weights_only=False)

    print(f"Downsampling neighbors for {split} set...")
    dataset = downsample_neighbors(dataset,use_fixed_n,n_keep)
    fsave = f"{split}_resampled_10_nei_from_0.05.pt"
    save_path = p_cache / fsave
    torch.save(dataset, save_path)
    print(f"✅ Saved {split} set → {save_path}")
