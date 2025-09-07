from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import sklearn.neighbors


@dataclass
class NeighborIndex:
    train_coords: torch.Tensor  # (N, 2)
    k: int

    def __post_init__(self):
        self.knn = sklearn.neighbors.NearestNeighbors(n_neighbors=self.k + 1)
        self.knn.fit(self.train_coords.numpy())

    def neighbors_for(self, coord: torch.Tensor, exclude_coord: Optional[torch.Tensor] = None) -> np.ndarray:
        """Return indices of k nearest neighbors in the training set.
        If exclude_coord is provided and equals some train point, drop it.
        """
        c = coord.detach().cpu().numpy().reshape(1, -1)
        dists, inds = self.knn.kneighbors(c, return_distance=True)
        inds = inds[0]  # (k+1,)
        if exclude_coord is not None:
            # if the closest is the point itself, drop it
            # (distance ~ 0 and coords equal)
            self_c = exclude_coord.detach().cpu().numpy().reshape(1, -1)
            d0 = np.linalg.norm(self.train_coords[inds[0]].numpy() - self_c)
            if d0 < 1e-12:
                inds = inds[1:]
            else:
                inds = inds[: self.k]
        else:
            inds = inds[: self.k]
        return inds


class TransformerPointDataset(Dataset):
    """
    Builds (memory_tokens, padding_mask, target_y_norm) per target point.
    - Coordinates are normalized relative to the CLS point: Δlat = lat_i - lat_cls, Δlon = lon_i - lon_cls
    - Δlat, Δlon are further standardized by train std (per-dimension)
    - Memory tokens include: [Δlat, Δlon, y_known_norm, is_observed]
    - CLS has no token in memory; the decoder's tgt is the learnable CLS token.
    """

    def __init__(
        self,
        targetset,  # SpatialDataset for targets (train/valid/test)
        trainset,   # SpatialDataset providing observed neighbors
        neighbor_index: NeighborIndex,
        k: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.target_coords = targetset.coords.to(device)
        self.target_y = targetset.y.to(device)  # normalized already by dt2_data
        self.train_coords = trainset.coords.to(device)
        self.train_y = trainset.y.to(device)    # normalized labels
        self.nei = neighbor_index
        self.k = k
        self.device = device

        # Stats for Δcoord standardization
        self.lat_std = trainset.coords[:, 0].std().item() + 1e-6
        self.lon_std = trainset.coords[:, 1].std().item() + 1e-6

        # Save label stats for de-normalization in evaluation if needed
        self.y_mean = getattr(trainset, "y_mean", None)
        self.y_std = getattr(trainset, "y_std", None)

        # Flag to indicate exclusion during training
        self.is_training_targets = targetset is trainset

    def __len__(self):
        return self.target_coords.shape[0]

    def _build_memory(self, cls_coord: torch.Tensor, exclude_self: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        # Neighbor indices from the training set
        inds = self.nei.neighbors_for(cls_coord, exclude_coord=cls_coord if exclude_self else None)
        mem_coords = self.train_coords[inds]  # (S, 2)
        mem_y = self.train_y[inds]            # (S, 1)

        # Relative coords
        dlat = (mem_coords[:, 0] - cls_coord[0]) / self.lat_std
        dlon = (mem_coords[:, 1] - cls_coord[1]) / self.lon_std

        # observed flag (all ones for real neighbors)
        is_obs = torch.ones_like(dlat)

        mem_tokens = torch.stack([dlat, dlon, mem_y.squeeze(-1), is_obs], dim=-1)  # (S, 4)
        pad_mask = torch.zeros(mem_tokens.size(0), dtype=torch.bool, device=mem_tokens.device)  # False = real
        return mem_tokens, pad_mask

    def __getitem__(self, idx: int):
        cls_coord = self.target_coords[idx]  # (2,)
        y_tgt = self.target_y[idx].squeeze(-1)  # scalar (normalized)

        exclude = self.is_training_targets  # during training, exclude self if it is in trainset
        mem_tokens, pad_mask = self._build_memory(cls_coord, exclude_self=exclude)

        return {
            "mem_tokens": mem_tokens,        # (S, 4)
            "pad_mask": pad_mask,            # (S,)
            "y": y_tgt,                      # ()
            "cls_coord": cls_coord,          # (2,)
        }


def collate_point_batches(batch):
    # Find max S in this mini-batch and left-pad to that length
    S = max(item["mem_tokens"].shape[0] for item in batch)
    B = len(batch)
    device = batch[0]["mem_tokens"].device

    mem_tokens = torch.zeros(B, S, batch[0]["mem_tokens"].shape[1], device=device)
    pad_mask = torch.ones(B, S, dtype=torch.bool, device=device)  # True = PAD by default
    y = torch.zeros(B, device=device)
    cls_coords = torch.zeros(B, 2, device=device)

    for i, item in enumerate(batch):
        s = item["mem_tokens"].shape[0]
        mem_tokens[i, :s] = item["mem_tokens"]
        pad_mask[i, :s] = item["pad_mask"]  # False for real tokens
        y[i] = item["y"]
        cls_coords[i] = item["cls_coord"]

    return mem_tokens, pad_mask, y, cls_coords
