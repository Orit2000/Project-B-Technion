from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error
import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
#mse = mean_squared_error(targets, predictions)
#matplotlib.use("TkAgg")  # Force GUI backend

# Accuracy Metrics
def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
#def MAPE(y_true, y_pred):
    
    
def MAP(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred))/len(y_true)

trainset = torch.load(r"./cache/trainset_n32_e035_1arc_v3_cropped_k50_keep_n0.00125.pt",weights_only=False)
validset = torch.load(r"./cache/validset_n32_e035_1arc_v3_cropped_k50_keep_n0.00125.pt",weights_only=False)
testset = torch.load(r"./cache/testset_n32_e035_1arc_v3_cropped_k50_keep_n0.00125.pt",weights_only=False)

# Convert tensors to numpy
train_coords = trainset.coords.detach().cpu().numpy()
train_y = (trainset.y * trainset.y_std + trainset.y_mean).detach().cpu().numpy().flatten()

valid_coords = validset.coords.detach().cpu().numpy()
valid_y_true = (validset.y * validset.y_std + validset.y_mean).detach().cpu().numpy().flatten()

test_coords = testset.coords.detach().cpu().numpy()
test_y_true = (testset.y * trainset.y_std + trainset.y_mean).detach().cpu().numpy().flatten()

# Perform interpolation: options = 'nearest', 'linear', 'cubic'
interp_method = 'nearest'
valid_y_interp = griddata(train_coords, train_y, valid_coords,method=interp_method)
test_y_interp = griddata(train_coords, train_y, test_coords,method=interp_method)

# Compute interpolation error
# Valid
interp_map_valid = MAP(valid_y_true, valid_y_interp)
interp_mse_valid = MSE(valid_y_true, valid_y_interp)
print(f"mse on valid: {interp_mse_valid}")
print(f"map on valid: {interp_map_valid}")

# Test
interp_map_test = MAP(test_y_true, test_y_interp)
interp_mse_test = MSE(test_y_true, test_y_interp)
print(f"mse on test: {interp_mse_test}")
print(f"map on test: {interp_map_test}")

#
test_error = test_y_true - test_y_interp
valid_error = valid_y_true - valid_y_interp

# Plot error map
plt.figure(1)
sc = plt.scatter(test_coords[:, 1], test_coords[:, 0], c=test_error, cmap="coolwarm", s=5)
plt.colorbar(sc, label="Interpolation Error (m)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Interpolation Error Map Test ({interp_method})")
plt.grid(True)
plt.savefig("basic_interp_reference/error_on_map_test.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot error map
plt.figure(2)
sc = plt.scatter(valid_coords[:, 1], valid_coords[:, 0], c=valid_error, cmap="coolwarm", s=5)
plt.colorbar(sc, label="Interpolation Error (m)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Interpolation Error Map Valid ({interp_method})")
plt.grid(True)
plt.savefig("basic_interp_reference/error_on_map_valis.png", dpi=300, bbox_inches='tight')
plt.show()

# Histogram
plt.figure(figsize=(10, 8))
plt.hist(test_error, bins=75, color='gray')
plt.title(f"Error Test Histogram - {interp_method} Interpolation")
plt.xlabel("Error (m)")
plt.ylabel("Amount")
plt.grid(True)
plt.savefig("basic_interp_reference/Error_Test_Histogram.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 8))
plt.hist(valid_error, bins=75, color='gray')
plt.title(f"Error Valid Histogram - {interp_method} Interpolation")
plt.xlabel("Error(m)")
plt.ylabel("Amount")
plt.grid(True)
plt.savefig("basic_interp_reference/Error_Valid_Histogram.png", dpi=300, bbox_inches='tight')
plt.show()

