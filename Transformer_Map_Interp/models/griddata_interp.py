from scipy.interpolate import griddata
import numpy as np
import torch
import sys, os
sys.path.append(os.path.abspath("."))  # project root
from Transformer_Map_Interp.datasets.data import SpatialDataset
from Transformer_Map_Interp.datasets.transformerRegressorDataClass import TransformerPointDataset
from scipy.spatial import QhullError
# def loo_griddata(coords, y, method="linear"):
#     preds = np.empty_like(y)
#     for i in range(len(coords)):
#         mask = np.ones(len(coords), dtype=bool)
#         mask[i] = False  # exclude self
#         pred = griddata(coords[mask], y[mask], coords[i], method=method)
#         if pred is not None and np.ndim(pred) > 0:
#             pred = pred.item()  # extract scalar
#         mask = np.isnan(pred)
#         if np.any(mask):
#             pred[mask] = griddata(coords[mask], y[mask], coords[i], method="nearest")
#         preds[i] = pred
        
#     return preds


def loo_griddata(coords, y, method="linear", fill_with_nearest=True):
    coords = np.asarray(coords)
    y = np.asarray(y).reshape(-1)
    assert coords.ndim == 2 and coords.shape[0] == y.shape[0], "shape mismatch"

    preds = np.empty(y.shape[0], dtype=float)

    for i in range(coords.shape[0]):
        # exclude self
        idx_mask = np.ones(coords.shape[0], dtype=bool)
        idx_mask[i] = False

        try:
            tri_pred = griddata(coords[idx_mask], y[idx_mask], coords[i], method=method)
        except QhullError:
            # triangulation can fail with too few/degenerate points; fallback
            tri_pred = np.nan

        # extract scalar if it’s a 0-dim array
        if tri_pred is not None and np.ndim(tri_pred) > 0:
            tri_pred = tri_pred.item()

        # optional: fill extrapolation/failed cases with nearest
        if fill_with_nearest and (tri_pred is None or np.isnan(tri_pred)):
            nn_pred = griddata(coords[idx_mask], y[idx_mask], coords[i], method="nearest")
            tri_pred = nn_pred.item() if np.ndim(nn_pred) > 0 else nn_pred

        preds[i] = tri_pred

    return preds

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def MAP(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred))/len(y_true)


trainset = torch.load(r"./Transformer_Map_Interp/cache/trainset_n32_e035_1arc_v3_cropped_train_n32_e035_1arc_v3_cropped_val_n32_e035_1arc_v3_cropped_test_keep_n0.001_seed5.pt",map_location="cpu",weights_only=False)
validset = torch.load(r"./Transformer_Map_Interp/cache/validset_n32_e035_1arc_v3_cropped_train_n32_e035_1arc_v3_cropped_val_n32_e035_1arc_v3_cropped_test_keep_n0.001_seed5.pt",map_location="cpu",weights_only=False)
testset = torch.load(r"./Transformer_Map_Interp/cache/testset_n32_e035_1arc_v3_cropped_train_n32_e035_1arc_v3_cropped_val_n32_e035_1arc_v3_cropped_test_keep_n0.001_seed5.pt",map_location="cpu",weights_only=False)


train_coords = trainset.coords.detach().cpu().numpy()
train_y = trainset.y.detach().cpu().numpy().flatten()
train_y_norm = trainset.y_norm.detach().cpu().numpy().flatten()

valid_coords = validset.coords.detach().cpu().numpy()
valid_y_true = validset.y.detach().cpu().numpy().flatten()

val_coords = np.concatenate([train_coords, valid_coords])
val_y = np.concatenate([train_y, valid_y_true])

# Predict only for the validation subset (not for training)
start_idx = len(train_coords)
print("Performing LOO griddata interpolation on validation set...")
val_pred = loo_griddata(val_coords, val_y, method="linear")[start_idx:]

interp_map_valid = MAP(valid_y_true, val_pred)
interp_mse_valid = MSE(valid_y_true, val_pred)
print(f"mse on valid: {interp_mse_valid}")
print(f"map on valid: {interp_map_valid}")
