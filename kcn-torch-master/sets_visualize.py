import numpy as np
import pandas as pd
from data import SpatialDataset
from matplotlib import pyplot as plt
import dt2_data
import argument
import experiment
import torch


# Load sets (adjust paths if needed)
torch.serialization.add_safe_globals([SpatialDataset])
trainset = torch.load("cache/trainset_n32_e035_1arc_v3_cropped_k50_keep_n0.005.pt", map_location="cpu")
validset = torch.load("cache/validset_n32_e035_1arc_v3_cropped_k50_keep_n0.005.pt", map_location="cpu")
testset = torch.load("cache/testset_n32_e035_1arc_v3_cropped_k50_keep_n0.005.pt", map_location="cpu")

# Extract coordinates
train_coords = trainset.coords.numpy()
valid_coords = validset.coords.numpy()
test_coords = testset.coords.numpy()

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(train_coords[:, 1], train_coords[:, 0], s=10, label='Train', alpha=0.6)
plt.scatter(valid_coords[:, 1], valid_coords[:, 0], s=10, label='Validation', alpha=0.6)
#plt.scatter(test_coords[:, 1], test_coords[:, 0], s=10, label='Test', alpha=0.6)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Train / Validation / Test Point Distribution")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.savefig("train_val_test_distribution.png")
plt.show()


# Extract elevation labels (y values)
train_elev = trainset.y.numpy().flatten()
valid_elev = validset.y.numpy().flatten()
test_elev = testset.y.numpy().flatten()
# Combine all elevation values to find global min/max
all_elev = np.concatenate([train_elev, valid_elev])  # + test_elev if you want

# Define shared bin edges
num_bins = 50
bin_edges = np.linspace(all_elev.min(), all_elev.max(), num_bins + 1)

# Plot histograms
plt.figure(figsize=(10, 6))
plt.hist(valid_elev, bins=bin_edges, alpha=0.6, label='Validation', color='orange')
plt.hist(train_elev, bins=bin_edges, alpha=0.6, label='Train', color='blue')
#plt.hist(test_elev, bins=50, alpha=0.6, label='Test', color='red')

plt.xlabel("Elevation (m)")
plt.ylabel("Amount")
plt.title("Elevation Distribution per Set")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()