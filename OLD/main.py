import numpy as np
import torch
from argument import parse_opt 
from experiment_with_cp import run_kcn
import numpy as np
import pandas as pd
from data import SpatialDataset
from matplotlib import pyplot as plt
import dt2_data
import argument
import torch


# %%
args = argument.parse_opt()
args.keep_n = 0.005/4
args.form_input_graph = 'mine'
args.model = 'kcn_gat'
print(args.dataset)
print(args.n_neighbors)
args.dataset = "n32_e035_1arc_v3_cropped"
print(args.dataset)
#test_error, test_preds, testset, epoch_valid_loss, epoch_valid_error, epoch_valid_mse, epoch_valid_mae, epoch_train_loss, epoch_train_error, epoch_train_mse, epoch_train_mae = run_kcn(args)
test_error, test_preds, testset, epoch_valid_loss, epoch_valid_error, epoch_valid_mse, epoch_valid_mae, epoch_train_loss, epoch_train_error, epoch_train_mse, epoch_train_mae, coverage_rate_tot, avg_interval_length_tot = run_kcn(args)
epochs = list(range(len(epoch_valid_loss)))

plt.figure(figsize=(10, 6))
plt.plot(epochs, epoch_train_loss, label='Train Loss', marker='o')
plt.plot(epochs, epoch_valid_loss, label='Validation Loss', marker='x')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(epochs, epoch_train_error, label='Train Loss', marker='o')
# plt.plot(epochs, epoch_valid_error, label='Validation Loss', marker='x')
# plt.xlabel("Epoch")
# plt.ylabel("Error: Estimated - True")
# plt.title("Training and Validation Error: Estimated - True")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(epochs, epoch_train_mse, label='Train Loss', marker='o')
plt.plot(epochs, epoch_valid_mse, label='Validation Loss', marker='x')
plt.xlabel("Epoch")
plt.ylabel("Self MSE Defnition")
plt.title("Training and Validation Self MSE Defnition")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(epochs, epoch_train_mae, label='Train Loss', marker='o')
plt.plot(epochs, epoch_valid_mae, label='Validation Loss', marker='x')
plt.xlabel("Epoch")
plt.ylabel("Self MAP Defnition")
plt.title("Training and Validation Self MAP Defnition")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot
plt.figure(figsize=(10, 8))
sc = plt.scatter(testset.coords[:, 1].numpy(), testset.coords[:, 0].numpy(), c=test_error, cmap="coolwarm", s=5)
plt.colorbar(sc, label="Prediction Error (m)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution of Prediction Errors")
plt.grid(True)
plt.show()


plt.hist(test_error, bins=100, color='gray')
plt.title("Prediction Error Histogram")
plt.xlabel("Error (Prediction - True)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# %%

# # Compute error
# errors = abs(testset.y*y_std + y_mean  - test_preds)

# # Plot
# plt.figure(figsize=(10, 8))
# sc = plt.scatter(testset.coords[:, 1], testset.coords[:, 0], c=errors, cmap="coolwarm", s=5)
# plt.colorbar(sc, label="Prediction Error (m)")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title("Spatial Distribution of Prediction Errors")
# plt.grid(True)
plt.show()

# %%


# %% Shoing the train, test, val sets
# # Load sets (adjust paths if needed)
# torch.serialization.add_safe_globals([SpatialDataset])
# trainset = torch.load("cache/trainset_n32_e035_1arc_v3_cropped_k50_keep_n0.01.pt", map_location="cpu")
# validset = torch.load("cache/validset_n32_e035_1arc_v3_cropped_k50_keep_n0.01.pt", map_location="cpu")
# testset = torch.load("cache/testset_n32_e035_1arc_v3_cropped_k50_keep_n0.01.pt", map_location="cpu")

# # Extract coordinates
# train_coords = trainset.coords.numpy()
# valid_coords = validset.coords.numpy()
# test_coords = testset.coords.numpy()

# # Plot
# plt.figure(figsize=(10, 8))
# plt.scatter(train_coords[:, 1], train_coords[:, 0], s=10, label='Train', alpha=0.6)
# plt.scatter(valid_coords[:, 1], valid_coords[:, 0], s=10, label='Validation', alpha=0.6)
# #plt.scatter(test_coords[:, 1], test_coords[:, 0], s=10, label='Test', alpha=0.6)
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title("Train / Validation / Test Point Distribution")
# plt.legend()
# plt.grid(True)
# plt.axis("equal")
# plt.tight_layout()
# plt.savefig("train_val_test_distribution.png")
# plt.show()


# # Extract elevation labels (y values)
# train_elev = trainset.y.numpy().flatten()
# valid_elev = validset.y.numpy().flatten()
# test_elev = testset.y.numpy().flatten()

# # Plot histograms
# plt.figure(figsize=(10, 6))
# plt.hist(valid_elev, bins=50, alpha=0.6, label='Validation', color='orange')
# plt.hist(train_elev, bins=50, alpha=0.6, label='Train', color='blue')
# plt.hist(test_elev, bins=50, alpha=0.6, label='Test', color='red')

# plt.xlabel("Elevation (m)")
# plt.ylabel("Amount")
# plt.title("Elevation Distribution per Set")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()