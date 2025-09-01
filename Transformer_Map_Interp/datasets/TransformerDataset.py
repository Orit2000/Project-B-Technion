import torch
from torch.utils.data import Dataset
import numpy as np

class TransformerDataset(Dataset):
    def __init__(self, spatial_dataset, args, max_obs_per_sample=None):
        self.dataset = spatial_dataset
        self.max_obs = max_obs_per_sample
        self.use_features = args.sf_use_fourier_feats
        self.include_obs_y = args.include_elevation_in_features

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # === Extract sample ===
        query_coord = self.dataset.query_coords[idx]        # (2,)
        print(f"query_coord shape: {query_coord.shape}\n")
        query_y     = self.dataset.query_y[idx]             # (1,)
        print(f"query_y shape: {query_y.shape}\n")
        obs_coords  = self.dataset.obs_coords[idx]          # (N_o, 2)
        print(f"obs_coords shape: {obs_coords.shape}\n")
        obs_y       = self.dataset.obs_y[idx]               # (N_o, 1)
        print(f"obs_y shape: {obs_y.shape}\n")

        # === Optional truncation ===
        if self.max_obs is not None and obs_coords.shape[0] > self.max_obs:
            sel = np.random.choice(obs_coords.shape[0], self.max_obs, replace=False)
            obs_coords = obs_coords[sel]
            obs_y = obs_y[sel]

        # === Combine tokens ===
        all_coords = torch.cat([obs_coords, query_coord.unsqueeze(0)], dim=0)  # (N+1, 2)
        print(f"all_coords shape: {all_coords.shape}\n")
        all_y = torch.cat([obs_y, query_y.unsqueeze(0)], dim=0)                # (N+1, 1)
        print(f"all_y shape: {all_y.shape}\n")

        # === Features ===
        if self.use_features:
            obs_feats = self.dataset.features[idx][:obs_coords.shape[0]]       # (N_o, d)
            query_feat = self.dataset.features[idx][-1].unsqueeze(0)           # (1, d)
            all_feats = torch.cat([obs_feats, query_feat], dim=0)             # (N+1, d)
        else:
            all_feats = None

        # === Masks ===
        N = all_coords.shape[0]
        obs_mask = torch.zeros(N, dtype=torch.bool)
        obs_mask[:-1] = True
        print(f"obs_mask shape: {obs_mask.shape}\n")

        query_mask = torch.zeros(N, dtype=torch.bool)
        query_mask[-1] = True
        print(f"query_mask shape: {query_mask.shape}\n")

        pad_mask = torch.zeros(N, dtype=torch.bool)  # no padding yet

        # Optionally append elevation to features
        if self.include_obs_y:
            if all_feats is None:
                all_feats = all_y
            else:
                all_feats = torch.cat([all_feats, all_y.unsqueeze(-1)], dim=1)

        return all_coords, all_feats, all_y, obs_mask, query_mask, pad_mask
