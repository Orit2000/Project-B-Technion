from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error
import numpy as np
import torch
from matplotlib import pyplot as plt

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2/len(y_true))

trainset = torch.load(r"cache\trainset_n32_e035_1arc_v3_cropped_k50_keep_n0.005.pt",weights_only=False)
testset = torch.load(r"cache\testset_n32_e035_1arc_v3_cropped_k50_keep_n0.005.pt",weights_only=False)

# Convert tensors to numpy
train_coords = trainset.coords.detach().cpu().numpy()
train_y = (trainset.y * trainset.y_std + trainset.y_mean).detach().cpu().numpy().flatten()

test_coords = testset.coords.detach().cpu().numpy()
test_y_true = (testset.y * trainset.y_std + trainset.y_mean).detach().cpu().numpy().flatten()

# Perform interpolation: options = 'nearest', 'linear', 'cubic'
interp_method = 'nearest'
test_y_interp = griddata(train_coords, train_y, test_coords,method=interp_method)

# Where interpolation fails (NaNs), fall back to nearest neighbor
# nan_mask = np.isnan(test_y_interp)
# if np.any(nan_mask):
#     test_y_interp[nan_mask] = griddata(test_coords[nan_mask],train_coords, train_y, method='nearest')

# Compute interpolation error
interp_errors = np.abs(test_y_true - test_y_interp)
intert_loss = mean_squared_error(test_y_true, test_y_interp)
print(np.mean(interp_errors))
print(np.std(interp_errors))
print(intert_loss)


# Plot error map
plt.figure(figsize=(10, 8))
sc = plt.scatter(test_coords[:, 1], test_coords[:, 0], c=interp_errors, cmap="coolwarm", s=5)
plt.colorbar(sc, label="Interpolation Error (m)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Interpolation Error Map ({interp_method})")
plt.grid(True)
plt.show()

# Histogram
plt.hist(interp_errors, bins=100, color='gray')
plt.title(f"Error Histogram - {interp_method} Interpolation")
plt.xlabel("Error (m)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
