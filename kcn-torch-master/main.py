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
import experiment
import torch

# # repeat the experiment in the paper
# def random_runs(args):
#     test_errors = []
#     for args.random_seed in range(10):
#         np.random.seed(args.random_seed)
#         torch.manual_seed(args.random_seed)
    
#         err = run_kcn(args)
#         test_errors.append(err)
    
#     test_errors = np.array(test_errors)
#     return test_errors
 
# if __name__ == "__main__":

#     args = parse_opt()
#     print(args)

#     # set random seeds
#     np.random.seed(args.random_seed)
#     torch.manual_seed(args.random_seed)
    
#     # run experiment on one train-test split
#     err = run_kcn(args)
#     print('Model: {}, test error: {}\n'.format(args.model, err))
    
    
    ## run all experiments on one dataset
    #model_error = dict()
    #for args.model in ["kcn", "kcn_gat", "kcn_sage"]:
    #    test_errors = random_runs(args)
    #    model_error[args.model] = (np.mean(test_errors), np.std(test_errors))
    #    print(model_error)


# %%
args = argument.parse_opt()
args.keep_n = 0.005*10/10
print(args.dataset)
print(args.n_neighbors)
args.dataset = "n32_e035_1arc_v3_cropped"
print(args.dataset)
args.model = 'kcn'
test_error, test_preds, testset, y_mean, y_std = experiment.run_kcn(args)
print('Model: {}, test error: {}\n'.format(args.model, test_error))

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