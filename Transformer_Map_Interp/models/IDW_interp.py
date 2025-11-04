import numpy as np
from scipy.interpolate import griddata
import numpy as np
import torch
import sys, os
sys.path.append(os.path.abspath("."))  # project root
from Transformer_Map_Interp.datasets.data import SpatialDataset
from Transformer_Map_Interp.datasets.transformerRegressorDataClass import TransformerPointDataset
from scipy.interpolate import LinearNDInterpolator
def _to_numpy(x):
    # works for numpy, torch tensors, lists
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)
# def idw_local(nei_coords_list, nei_y_list, query_coords, k=None, p=2.0, eps=1e-12):
#     """
#     Local IDW using precomputed neighbors per target.
#     - nei_coords_list: list of (S_i, 2) coords (np or torch)
#     - nei_y_list:      list of (S_i,) or (S_i,1) values (np or torch)
#     - query_coords:    (N,2) target coords (np or torch)
#     """
#     N = len(nei_coords_list)
#     preds = np.zeros(N, dtype=float)
#     qxy = _to_numpy(query_coords)

#     for i in range(N):
#         nei_xy = _to_numpy(nei_coords_list[i]).astype(float)     # (S,2)
#         nei_y  = _to_numpy(nei_y_list[i]).astype(float).reshape(-1)  # (S,)

#         if nei_xy.size == 0:
#             preds[i] = np.nan
#             continue

#         # distances to this query point
#         d = np.linalg.norm(nei_xy - qxy[i], axis=1)  # (S,)

#         # if any neighbor sits exactly at the query, return its y (no divide-by-zero)
#         # zero = (d < 1e-12)
#         # if np.any(zero):
#         #     # if multiple exact overlaps, average them (rare)
#         #     preds[i] = float(nei_y[zero].mean())
#         #     continue

#         # # optionally keep only the closest k
#         # if k is not None and len(d) > k:
#         #     idx = np.argpartition(d, k)[:k]
#         #     d = d[idx]; nei_y = nei_y[idx]

#         w = 1.0 / (d**p + eps)              # (S,)
#         preds[i] = float(np.sum(w * nei_y) / np.sum(w))

#     return preds
import numpy as np

def _to_numpy(x):
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _scalar(x):
    if x is None:
        return np.nan
    a = np.asarray(x)
    if a.size == 0:
        return np.nan
    # robustly take the first element (handles (), (1,), (1,1), etc.)
    return float(a.ravel()[0])

def idw_local_all(nei_coords_list, nei_y_list, query_coords, p=2.0, eps=1e-12):
    """
    Local IDW using *all* precomputed neighbors for each target point.
    """
    N = len(nei_coords_list)
    preds = np.zeros(N, dtype=float)
    qxy = _to_numpy(query_coords)

    for i in range(N):
        nei_xy = _to_numpy(nei_coords_list[i]).astype(float)
        nei_y  = _to_numpy(nei_y_list[i]).astype(float).reshape(-1)
        print(nei_xy.size)
        if nei_xy.size == 0:
            preds[i] = np.nan
            continue

        d = np.linalg.norm(nei_xy - qxy[i], axis=1)  # (S,)

        # exact match protection
        if np.any(d < 1e-12):
            preds[i] = float(nei_y[d.argmin()])
            continue

        w = 1.0 / (d**p + eps)
        preds[i] = float(np.sum(w * nei_y) / np.sum(w))

    return preds

def linear_local(nei_coords_list, nei_y_list, query_coords, fill_with_idw=True, p=2.0):
    """
    Local linear interpolation using precomputed neighbors.
    - nei_coords_list: list of (S_i, 2)
    - nei_y_list: list of (S_i,)
    - query_coords: (N, 2)
    - fill_with_idw: if True, fill NaN/extrapolation with local IDW
    """
    N = len(nei_coords_list)
    preds = np.zeros(N, dtype=float)
    qxy = _to_numpy(query_coords)

    for i in range(N):
        nei_xy = _to_numpy(nei_coords_list[i]).astype(float)
        nei_y  = _to_numpy(nei_y_list[i]).astype(float).reshape(-1)

        if len(nei_xy) < 3:
            # not enough points to form a triangle; fallback to mean
            preds[i] = np.mean(nei_y)
            continue

        try:
            interp = LinearNDInterpolator(nei_xy, nei_y, fill_value=np.nan)
            pred = interp(qxy[i])
            if np.isnan(pred) and fill_with_idw:
                # fallback to local IDW if outside convex hull
                d = np.linalg.norm(nei_xy - qxy[i], axis=1)
                w = 1.0 / (d**p + 1e-12)
                pred = np.sum(w * nei_y) / np.sum(w)
        except Exception:
            # triangulation sometimes fails if neighbors are colinear
            d = np.linalg.norm(nei_xy - qxy[i], axis=1)
            w = 1.0 / (d**p + 1e-12)
            pred = np.sum(w * nei_y) / np.sum(w)

        preds[i] = _scalar(pred)
    return preds
from scipy.interpolate import Rbf

def rbf_local(nei_coords_list, nei_y_list, query_coords, function='linear'):
    preds = np.zeros(len(nei_coords_list))
    qxy = _to_numpy(query_coords)
    for i, (xy, y) in enumerate(zip(nei_coords_list, nei_y_list)):
        xy = _to_numpy(xy); y = _to_numpy(y).reshape(-1)
        if len(xy) < 3:
            preds[i] = np.mean(y)
            continue
        try:
            rbf = Rbf(xy[:,0], xy[:,1], y, function=function)
            preds[i] = float(rbf(qxy[i,0], qxy[i,1]))
        except Exception:
            preds[i] = np.mean(y)
    return preds

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def MAP(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred))/len(y_true)


trainset = torch.load(r"./Transformer_Map_Interp/cache/trainset_n32_e035_1arc_v3_cropped_train_n32_e035_1arc_v3_cropped_val_n32_e035_1arc_v3_cropped_test_keep_n0.001_seed5.pt",map_location="cpu",weights_only=False)
validset = torch.load(r"./Transformer_Map_Interp/cache/validset_n32_e035_1arc_v3_cropped_train_n32_e035_1arc_v3_cropped_val_n32_e035_1arc_v3_cropped_test_keep_n0.001_seed5.pt",map_location="cpu",weights_only=False)
testset = torch.load(r"./Transformer_Map_Interp/cache/testset_n32_e035_1arc_v3_cropped_train_n32_e035_1arc_v3_cropped_val_n32_e035_1arc_v3_cropped_test_keep_n0.001_seed5.pt",map_location="cpu",weights_only=False)

print("Performing Local IDW interpolation on validation and test sets...")
# val_pred = idw_local_all(validset.obs_coords, validset.obs_y,
#                      validset.query_coords)
# test_pred = idw_local_all(testset.obs_coords, testset.obs_y,
#                       testset.query_coords)

# mse_val = np.mean((val_pred - validset.query_y.detach().cpu().numpy())**2)
# mae_val = np.mean(np.abs(val_pred - validset.query_y.detach().cpu().numpy()))
# mse_test = np.mean((test_pred - testset.query_y.detach().cpu().numpy())**2)
# mae_test = np.mean(np.abs(test_pred - testset.query_y.detach().cpu().numpy()))
# print(f"[Local IDW] Validation MSE={mse_val:.3f}, MAE={mae_val:.3f}")
# print(f"[Local IDW] Test MSE={mse_test:.3f}, MAE={mae_test:.3f}")

# val_pred_lin = linear_local(validset.obs_coords, validset.obs_y, validset.query_coords)
# test_pred_lin = linear_local(testset.obs_coords, testset.obs_y, testset.query_coords)

y_true_val = validset.q_y_norm.detach().cpu().numpy()
# mse_val_lin = np.mean((val_pred_lin - y_true_val)**2)
# mae_val_lin = np.mean(np.abs(val_pred_lin - y_true_val))
# print(f"[Local LinearND] Validation MSE={mse_val_lin:.3f}, MAE={mae_val_lin:.3f}")
y_true_test = testset.q_y_norm.detach().cpu().numpy()
# mse_test_lin = np.mean((test_pred_lin - y_true_test)**2)
# mae_test_lin = np.mean(np.abs(test_pred_lin - y_true_test))
# print(f"[Local LinearND] Test MSE={mse_test_lin:.3f}, MAE={mae_test_lin:.3f}")

val_pred_rbf = rbf_local(validset.obs_coords, validset.obs_y_norm, validset.query_coords)
test_pred_rbf = rbf_local(testset.obs_coords, testset.obs_y_norm, testset.query_coords)
mse_val_rbf = np.mean((val_pred_rbf - y_true_val)**2)
mae_val_rbf = np.mean(np.abs(val_pred_rbf - y_true_val))
mse_test_rbf = np.mean((test_pred_rbf - y_true_test)**2)
mae_test_rbf = np.mean(np.abs(test_pred_rbf - y_true_test))
print(f"[Local RBF] Validation MSE={mse_val_rbf:.3f}, MAE={mae_val_rbf:.3f}")
print(f"[Local RBF] Test MSE={mse_test_rbf:.3f}, MAE={mae_test_rbf:.3f}")   