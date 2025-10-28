import torch
import numpy as np
import sys, os
sys.path.append(os.path.abspath("."))  # project root
from Transformer_Map_Interp.datasets.dt2_data import DT2Dataset, set_creation_func

# === CONFIG ===
dt2_path = "Transformer_Map_Interp/datasets/n32_e035_1arc_v3_cropped_val.tiff"
trainset_path = "Transformer_Map_Interp/cache/trainset_n32_e035_1arc_v3_cropped_train_n32_e035_1arc_v3_cropped_val_n32_e035_1arc_v3_cropped_test_keep_n0.001_seed5.pt"
save_path = "Transformer_Map_Interp/cache/valset_recreated.pt"
max_radius_km = 0.75  # same as in your args
keep_ratio = 0.05    # same as before
random_seed = 5       # same seed as before

# === LOAD EXISTING TRAINSET ===
trainset = torch.load(trainset_path, map_location="cpu", weights_only=False)
print("Loaded trainset successfully!")

# === LOAD NEW TEST TILE ===
from Transformer_Map_Interp.datasets.dt2_data import DT2Dataset
dataset = DT2Dataset(dt2_path, include_elevation_in_features=False)
total = dataset.coords.shape[0]
keep_n = int(total * keep_ratio)
rng = np.random.RandomState(seed=random_seed)
selected_idx = rng.choice(total, size=keep_n, replace=False)
print(f"Selected {keep_n} / {total} points for test")

# === CREATE NEW TESTSET ===
from Transformer_Map_Interp.datasets.dt2_data import set_creation_func
testset = set_creation_func(dataset, selected_idx, max_radius_km, trainset=trainset,neighbors_train_only=False)

# === SAVE ===
torch.save(testset, save_path)
