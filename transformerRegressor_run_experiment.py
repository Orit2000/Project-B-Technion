from typing import Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Transformer_Map_Interp.datasets.transformerRegressorDataClass import NeighborIndex, TransformerPointDataset, collate_point_batches
from Transformer_Map_Interp.datasets.dt2_data import load_dt2_data # late import to reuse existing loader

from Transformer_Map_Interp.models.transformerRegressor import TransformerCLSRegressor


def run_transformer(args) -> Tuple[float, float]:
    """
    Trains and evaluates the decoder-only Transformer with a CLS token.
    Assumes dt2_data.load_dt2_data() produced normalized labels (y) and attached y_mean/y_std to trainset.
    """

    # 1) Load (and cache) DT2 sets in SpatialDataset form
    trainset, validset, testset, calibset = load_dt2_data(args)

    # 2) Build neighbor index over the training coords
    nei = NeighborIndex(train_coords=trainset.coords.cpu(), k=args.n_neighbors)

    # 3) Build per-point datasets with relative coordinates to CLS
    dev = args.device
    ds_train = TransformerPointDataset(trainset, trainset, nei, k=args.n_neighbors, device=dev)
    ds_valid = TransformerPointDataset(validset, trainset, nei, k=args.n_neighbors, device=dev)
    ds_test  = TransformerPointDataset(testset,  trainset, nei, k=args.n_neighbors, device=dev)

    # 4) Dataloaders
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_point_batches)
    valid_loader = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False, collate_fn=collate_point_batches)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_point_batches)

    # 5) Model
    in_dim = 4  # [Δlat, Δlon, y_known_norm, is_observed]
    model = TransformerCLSRegressor(
        in_dim=in_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        cls_init=args.cls_init,
        use_posenc=args.use_posenc,
    ).to(dev)

    # 6) Optim & loss
    loss_fn = torch.nn.MSELoss(reduction="mean")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    epoch_train_loss = []
    epoch_valid_loss = []

    for epoch in range(args.epochs):
        model.train()
        tl = 0.0
        n = 0
        for mem, mask, y, _ in tqdm(train_loader, desc=f"[Epoch {epoch}] train"):
            y_hat = model(mem, mask)
            loss = loss_fn(y_hat, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tl += loss.item() * y.size(0)
            n += y.size(0)
        tl /= max(n, 1)
        epoch_train_loss.append(tl)

        model.eval()
        with torch.no_grad():
            vl = 0.0
            n = 0
            for mem, mask, y, _ in tqdm(valid_loader, desc=f"[Epoch {epoch}] valid"):
                y_hat = model(mem, mask)
                loss = loss_fn(y_hat, y)
                vl += loss.item() * y.size(0)
                n += y.size(0)
            vl /= max(n, 1)
            epoch_valid_loss.append(vl)

        print(f"Epoch {epoch}: train MSE={tl:.6f} | valid MSE={vl:.6f}")

        # Early stop heuristic
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch > args.es_patience) and len(epoch_valid_loss) > (args.es_patience + 3):
            recent = sum(epoch_valid_loss[-3:]) / 3
            prev = sum(epoch_valid_loss[-(args.es_patience + 3):-3]) / args.es_patience
            if recent > prev:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # 7) Test (report in real meters)
    y_std = ds_train.y_std.to(dev) if ds_train.y_std is not None else torch.tensor(1.0, device=dev)
    y_mean = ds_train.y_mean.to(dev) if ds_train.y_mean is not None else torch.tensor(0.0, device=dev)

    def eval_loader(loader):
        model.eval()
        se = 0.0
        ae = 0.0
        n = 0
        with torch.no_grad():
            for mem, mask, y, _ in loader:
                y_hat = model(mem, mask)
                # de-normalize
                y_hat_real = y_hat * y_std + y_mean
                y_real = y * y_std + y_mean
                se += torch.sum((y_hat_real - y_real) ** 2).item()
                ae += torch.sum(torch.abs(y_hat_real - y_real)).item()
                n += y.size(0)
        mse = se / max(n, 1)
        mae = ae / max(n, 1)
        return mse, mae

    test_mse, test_mae = eval_loader(test_loader)
    print(f"TEST  MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")

    return test_mse, test_mae
