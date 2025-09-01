import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from datasets.data import SpatialDataset
#from Transformer_Map_Interp.datasets.data import SpatialDataset
#from ..datasets.dt2_data import load_dt2_data


# ========= CONFIG =========
DATASET_NAME = "n32_e035_1arc_v3_cropped"
CACHE_DIR = "./Transformer_Map_Interp/cache"
K = 50
KEEP_N = 0.005
SHOW_RADIUS_KM = 3.0  # Optional: for on-the-fly masking
EXAMPLE_INDICES = [0, 10, 20]  # Change to visualize more examples
# ==========================

# === Load .pt files ===
def load_cached_sets(dataset_name, k, keep_n, cache_dir="cache"):
    key = f"{dataset_name}_k{k}_keep_n{keep_n}"
    paths = {
        "train": os.path.join(cache_dir, f"trainset_{key}.pt"),
        "valid": os.path.join(cache_dir, f"validset_{key}.pt"),
        "test":  os.path.join(cache_dir, f"testset_{key}.pt"),
    }
    sets = {}
    for name, path in paths.items():
        if os.path.exists(path):
            sets[name] = torch.load(path,weights_only=False)
        else:
            print(f"[WARNING] Missing {path}")
    return sets

# === Haversine distance (for dynamic obs mask if needed) ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# === Plot set distribution ===
def plot_set_distributions(train, valid, test):
    plt.figure(figsize=(10, 8))
    if train:
        plt.scatter(train.coords[:, 1], train.coords[:, 0], s=8, c='blue', label="Train", alpha=0.6)
    if valid:
        plt.scatter(valid.coords[:, 1], valid.coords[:, 0], s=8, c='orange', label="Valid", alpha=0.6)
    if test:
        plt.scatter(test.coords[:, 1], test.coords[:, 0], s=8, c='red', label="Test", alpha=0.6)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.legend(); plt.grid(True); plt.axis("equal")
    plt.title("Coordinate Distribution Across Sets")
    plt.tight_layout(); plt.show()

# === Plot Transformer sample: query + observed ===
def plot_transformer_example(dataset, idx, train_coords, radius_km=SHOW_RADIUS_KM):
    q = dataset.coords[idx].numpy()
    t_coords = train_coords.numpy()
    #dists = haversine(q[0], q[1], t_coords[:, 0], t_coords[:, 1])
    #obs_mask = dists <= radius_km
    obs_coords = dataset.obs_coords[idx]

    plt.figure(figsize=(7, 6))
    plt.scatter(obs_coords[:, 1], obs_coords[:, 0], c='blue', s=12, label='Observed', alpha=0.7)
    plt.scatter(q[1], q[0], c='red', s=40, label='Query', edgecolor='k', linewidth=0.5)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.legend(); plt.grid(True); plt.axis("equal")
    plt.title(f"Transformer Input: Sample #{idx} (within {radius_km} km)")
    plt.tight_layout(); plt.show()

# === Run the visualizations ===
def main():
    sets = load_cached_sets(DATASET_NAME, K, KEEP_N, cache_dir=CACHE_DIR)
    train, valid, test = sets.get("train"), sets.get("valid"), sets.get("test")
    
    if not all([train, valid, test]):
        print("[ERROR] Missing one or more datasets.")
        return

    # (1) Plot the full set distribution
    plot_set_distributions(train, valid, test)

    # (2) Show examples from test set (query vs surrounding obs from train)
    for idx in EXAMPLE_INDICES:
        if idx < len(test.coords):
            plot_transformer_example(test, idx, train.coords)
        else:
            print(f"[SKIP] Index {idx} out of bounds for test set.")

if __name__ == "__main__":
    main()
