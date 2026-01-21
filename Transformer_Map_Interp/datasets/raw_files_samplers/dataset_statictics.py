import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("."))  # project root
from Transformer_Map_Interp.datasets.data import SpatialDataset
from Transformer_Map_Interp.datasets.transformerRegressorDataClass import TransformerPointDataset
from Transformer_Map_Interp.datasets.transformerRegressorDataClass import TransformerPointDataset, collate_point_batches
from Transformer_Map_Interp.datasets.dt2_data_orig____ import load_multi_dt2_data

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
cache_dir = "Transformer_Map_Interp/cache"  # path to cached npz/tensors
device = "cpu"
sets = ["trainset_n32_e035_1arc_v3_cropped_train_n32_e035_1arc_v3_cropped_val_n32_e035_1arc_v3_cropped_test_keep_n0.001_seed5", "validset_n32_e035_1arc_v3_cropped_train_n32_e035_1arc_v3_cropped_val_n32_e035_1arc_v3_cropped_test_keep_n0.001_seed5", "testset_n32_e035_1arc_v3_cropped_train_n32_e035_1arc_v3_cropped_val_n32_e035_1arc_v3_cropped_test_keep_n0.001_seed5"]

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
def describe_elevations(y, label):
    y = np.asarray(y)
    print(f"\n[{label}] Elevation Stats:")
    print(f"  Mean:  {y.mean():.3f}")
    print(f"  Std:   {y.std():.3f}")
    print(f"  Min:   {y.min():.3f}")
    print(f"  Max:   {y.max():.3f}")
    print(f"  NaNs:  {np.isnan(y).sum()}")

def describe_neighbors(ds, label):
    neighbor_counts = []
    for i in range(len(ds)):
        obs_coords = ds.obs_coords[i]
        neighbor_counts.append(obs_coords.shape[0])
    neighbor_counts = np.array(neighbor_counts)
    print(f"\n[{label}] Neighbor Count:")
    print(f"  Mean: {neighbor_counts.mean():.2f}")
    print(f"  Std:  {neighbor_counts.std():.2f}")
    print(f"  Min:  {neighbor_counts.min()}  Max: {neighbor_counts.max()}")
    return neighbor_counts

def plot_hist(data, title, xlabel, bins=40):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------
# Load dataset splits
# -------------------------------------------------------------
for phase in sets:
    print(f"\n--- Analyzing {phase.upper()} set ---")
    #dataset = TransformerPointDataset(cache_dir=cache_dir, split=phase, device=device)
    path = os.path.join(cache_dir, f"{phase}.pt")
    dataset = torch.load(path,map_location="cpu",weights_only=False)
    
    # Elevations
    obs_y = torch.cat(dataset.obs_y).cpu().numpy()  # unnormalized elevations
    q_y = torch.cat(dataset.y).cpu().numpy()
    obs_y_norm = torch.cat(dataset.obs_y_norm).cpu().numpy()  # unnormalized elevations
    q_y_norm = torch.cat(dataset.y_norm).cpu().numpy()
    

    # Print elevation stats
    describe_elevations(obs_y, f"{phase} (raw)")
    describe_elevations(y_norm, f"{phase} (normalized)")

    # Neighbor stats
    neighbor_counts = describe_neighbors(dataset, phase)

    # -------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------
    plot_hist(obs_y, f"{phase} Elevation (raw)", "Elevation [m]")
    plot_hist(y_norm, f"{phase} Elevation (normalized)", "Normalized elevation")
    plot_hist(neighbor_counts, f"{phase} Neighbor Count", "Neighbors per query")

    # Optional: show spatial coverage
    if hasattr(dataset, "obs_coords"):
        all_coords = torch.cat(dataset.obs_coords).cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.scatter(all_coords[:, 1], all_coords[:, 0], s=1, alpha=0.3)
        plt.title(f"{phase} observation coordinate map")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.show()

# -------------------------------------------------------------
# Overall summary (optional)
# -------------------------------------------------------------
print("\n✅ Dataset analysis complete.")
