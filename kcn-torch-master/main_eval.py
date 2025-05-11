import torch
import numpy as np
import torch
from argument import parse_opt 
from experiment import run_kcn
import numpy as np
import pandas as pd
from data import SpatialDataset
from matplotlib import pyplot as plt
import dt2_data
import argument
import kcn
args = argument.parse_opt()
args.keep_n = 0.005*10
print(args.dataset)
print(args.n_neighbors)
args.dataset = "n32_e035_1arc_v3_cropped"
print(args.dataset)
args.model = 'kcn'

trainset = torch.load(r"cache\trainset_n32_e035_1arc_v3_cropped_k50_keep_n0.05.pt",weights_only=False)
testset = torch.load(r"cache\testset_n32_e035_1arc_v3_cropped_k50_keep_n0.05.pt",weights_only=False)
#  g-y_mean = trainset.y.mean(dim=0, keepdim=True)
# y_std = trainset.y.std(dim=0, keepdim=True) + 1e-6
y_std = trainset.y_std
y_mean = trainset.y_mean
model = kcn.KCN(trainset, args).to(args.device)
model.load_state_dict(torch.load(r"saved_models/kcn_n32_e035_1arc_v3/best_model_epoch9.pt"))
model.eval()
test_preds = model(testset.coords, testset.features, 5) 
#test_error = loss_func(test_preds, testset.y.to(args.device))
test_preds = test_preds * y_std + y_mean

errors = abs(testset.y*y_std + y_mean  - test_preds)
print(np.mean(errors.detach().numpy()))
print(np.std(errors.detach().numpy()))

# Plot
plt.figure(figsize=(10, 8))
sc = plt.scatter(testset.coords[:, 1].numpy(), testset.coords[:, 0].numpy(), c=errors.detach().numpy(), cmap="coolwarm", s=5)
plt.colorbar(sc, label="Prediction Error (m)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution of Prediction Errors")
plt.grid(True)
plt.show()

# %%
plt.hist(errors.detach().numpy(), bins=100, color='gray')
plt.title("Prediction Error Histogram")
plt.xlabel("Error (Prediction - True)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

