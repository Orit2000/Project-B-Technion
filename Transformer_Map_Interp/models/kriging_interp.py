from scipy.interpolate import griddata, Rbf
from pykrige.ok import OrdinaryKriging
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys, os
from sklearn.neighbors import KDTree
sys.path.append(os.path.abspath("."))  # project root
from Transformer_Map_Interp.datasets.data import SpatialDataset
from Transformer_Map_Interp.datasets.transformerRegressorDataClass import TransformerPointDataset
def MSE(y_true, y_pred): return np.mean((y_true - y_pred)**2)
def MAE(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))


def local_kriging_predict(pool_coords, pool_y, query_coords, k=100,
                          variogram_model="spherical", nugget=1e-5):
    """
    Predicts values at query_coords using local Ordinary Kriging on the k nearest neighbors.
    This avoids huge NxN memory issues by fitting a tiny Kriging model per query.
    """
    kdt = KDTree(pool_coords, metric='euclidean')
    preds = np.zeros(len(query_coords))
    for i, q in enumerate(query_coords):
        # Find k nearest neighbors from the pool
        idx = kdt.query(q.reshape(1, -1), k=min(k, len(pool_coords)),
                        return_distance=False)[0]
        neigh = pool_coords[idx]
        vals  = pool_y[idx]
        try:
            OK = OrdinaryKriging(
                neigh[:,1], neigh[:,0], vals,
                variogram_model=variogram_model,
                verbose=False, enable_plotting=False,
                nugget=nugget,
                coordinates_type="euclidean"
            )
            pred, _ = OK.execute("points", [q[1]], [q[0]])
            preds[i] = pred
        except Exception as e:
            # If local fit fails (degenerate geometry), fallback to mean
            preds[i] = np.mean(vals)
    return np.array(preds)

from pykrige.ok import OrdinaryKriging
import numpy as np

def kriging_from_obs(dataset, variogram_model="spherical", nugget=1e-5):
    preds = []
    for i in range(len(dataset.query_coords)):
        obs_coords = dataset.obs_coords[i].numpy()
        obs_y = dataset.obs_y[i].numpy()
        q_coord = dataset.query_coords[i].numpy()
        x = obs_coords[:,1]
        y = obs_coords[:,0]
        try:
            OK = OrdinaryKriging(
                x, y, obs_y,
                variogram_model=variogram_model,
                verbose=False, enable_plotting=False,
                coordinates_type="euclidean"
            )
            pred, _ = OK.execute("points", [q_coord[1]], [q_coord[0]])
        except ValueError as e:
            print(f"Kriging failed (degenerate variogram): {e}")
            pred = np.mean(obs_y)
        preds.append(float(pred))
            
    return np.array(preds)



# ===========================================================
# Run all methods on VALID + TEST
# ===========================================================
print("loading trainset...")
trainset = torch.load(r"./Transformer_Map_Interp/cache/trainset_n32_e035_1arc_v3_cropped_train_n32_e035_1arc_v3_cropped_val_n32_e035_1arc_v3_cropped_test_keep_n0.001_seed5.pt",map_location="cpu",weights_only=False)
#validset = torch.load(r"./Transformer_Map_Interp/cache/validset_n32_e035_1arc_v3_cropped_train_n32_e035_1arc_v3_cropped_val_n32_e035_1arc_v3_cropped_test_keep_n0.001_seed5.pt",map_location="cpu",weights_only=False)
#testset = torch.load(r"./Transformer_Map_Interp/cache/testset_n32_e035_1arc_v3_cropped_train_n32_e035_1arc_v3_cropped_val_n32_e035_1arc_v3_cropped_test_keep_n0.001_seed5.pt",map_location="cpu",weights_only=False)
print("loading testset...")
testset = torch.load(r"./Transformer_Map_Interp/cache/testset_recreated.pt",map_location="cpu",weights_only=False)
print("loading valset...")
validset = torch.load(r"./Transformer_Map_Interp/cache/valset_recreated.pt",map_location="cpu",weights_only=False)
# Convert tensors to numpy
train_coords = trainset.coords.detach().cpu().numpy()
train_y = trainset.y.detach().cpu().numpy().flatten()
train_y_norm = trainset.y_norm.detach().cpu().numpy().flatten()

valid_coords = validset.coords.detach().cpu().numpy()
valid_y_true = validset.y.detach().cpu().numpy().flatten()
#valid_y_true_norm = validset.y_norm.detach().cpu().numpy().flatten()

test_coords = testset.coords.detach().cpu().numpy()
test_y_true = testset.y .detach().cpu().numpy().flatten()
#test_y_true = testset.y .detach().cpu().numpy().flatten()

val_pool_coords = np.concatenate([train_coords, valid_coords], axis=0)
val_pool_y = np.concatenate([train_y, valid_y_true], axis=0)

test_pool_coords = np.concatenate([train_coords, test_coords], axis=0)
test_pool_y = np.concatenate([train_y, test_y_true], axis=0)

# ===========================================================
# 3️⃣ Kriging on VALID (train + test pool)
# ===========================================================
# valid_y_krig = local_kriging_predict(val_pool_coords, val_pool_y, valid_coords, k=100)
# test_y_krig = local_kriging_predict(test_pool_coords, test_pool_y, test_coords, k=120)
valid_y_krig = kriging_from_obs(validset)
print(f"VALID: MSE={MSE(valid_y_true, valid_y_krig):.4f} | MAE={MAE(valid_y_true, valid_y_krig):.4f}")

test_y_krig = kriging_from_obs(testset)
print(f"TEST:  MSE={MSE(test_y_true, test_y_krig):.4f} | MAE={MAE(test_y_true, test_y_krig):.4f}")