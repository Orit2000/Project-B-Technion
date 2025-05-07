import numpy as np
import pandas as pd
from data import SpatialDataset
from matplotlib import pyplot as plt
import dt2_data
import argument
import experiment
import torch
import kcn 
# Model Results Evaluations
from types import SimpleNamespace

# Load metadata
metadata = torch.load("saved_models/kcn_n32_e035_1arc_v3/kcn_n32_e035_1arc_v3_cropped_metadata.pt")
args = metadata['args']  # must contain hyperparameters, device, etc.
print(args)
torch.serialization.add_safe_globals([SpatialDataset])
trainset = torch.load("cache/trainset_n32_e035_1arc_v3_cropped_k50_keep_n0.01.pt")
validset = torch.load("cache/validset_n32_e035_1arc_v3_cropped_k50_keep_n0.01.pt", map_location="cpu")
testset = torch.load("cache/testset_n32_e035_1arc_v3_cropped_k50_keep_n0.01.pt", map_location="cpu")

# Load model
args = SimpleNamespace(**args)
print(args)
model = kcn.KCN(trainset, args)
model.load_state_dict(torch.load("saved_models/kcn_n32_e035_1arc_v3/best_model_epoch38.pt", map_location=args.device))
model.eval()

full_coords = torch.cat([trainset.coords, validset.coords, testset.coords], dim=0)
full_features = torch.cat([trainset.features, validset.features, testset.features], dim=0)
full_labels = torch.cat([trainset.y, validset.y, testset.y], dim=0)

with torch.no_grad():
    preds = model(full_coords, full_features, args.top_k).cpu().numpy()
    true = full_labels.cpu().numpy()

# Combine coordinates and predictions
latlon_pred = pd.DataFrame({
    "lat": full_coords[:, 0].numpy(),
    "lon": full_coords[:, 1].numpy(),
    "pred": preds.flatten(),
    "true": true.flatten()
})

# Round to a grid if needed
latlon_pred = latlon_pred.sort_values(["lat", "lon"])

# Pivot to image grid
pivot_pred = latlon_pred.pivot(index="lat", columns="lon", values="pred")
pivot_true = latlon_pred.pivot(index="lat", columns="lon", values="true")

# Now plot
plt.imshow(pivot_true.values, cmap="terrain", origin="upper")
plt.title("GT Elevation Map")
plt.show()

plt.imshow(pivot_pred.values, cmap="terrain", origin="upper")
plt.title("Predicted Elevation Map")
plt.show()