# from typing import Tuple
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from Transformer_Map_Interp.datasets.transformerRegressorDataClass import NeighborIndex, TransformerPointDataset, collate_point_batches
# from Transformer_Map_Interp.datasets.dt2_data import load_dt2_data # late import to reuse existing loader

# from Transformer_Map_Interp.models.transformerRegressor import TransformerCLSRegressor


# def run_transformer(args) -> Tuple[float, float]:
#     """
#     Trains and evaluates the decoder-only Transformer with a CLS token.
#     Assumes dt2_data.load_dt2_data() produced normalized labels (y) and attached y_mean/y_std to trainset.
#     """

#     # 1) Load (and cache) DT2 sets in SpatialDataset form
#     trainset, validset, testset, calibset = load_dt2_data(args)

#     # 2) Build neighbor index over the training coords
#     nei = NeighborIndex(train_coords=trainset.coords.cpu(), k=args.n_neighbors)

#     # 3) Build per-point datasets with relative coordinates to CLS
#     dev = args.device
#     ds_train = TransformerPointDataset(trainset, trainset, nei, k=args.n_neighbors, device=dev)
#     ds_valid = TransformerPointDataset(validset, trainset, nei, k=args.n_neighbors, device=dev)
#     ds_test  = TransformerPointDataset(testset,  trainset, nei, k=args.n_neighbors, device=dev)

#     # 4) Dataloaders
#     train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_point_batches) #calls __getitem()__ function
#     valid_loader = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False, collate_fn=collate_point_batches)
#     test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_point_batches)

#     # 5) Model
#     in_dim = 4  # [Δlat, Δlon, y_known_norm, is_observed]
#     model = TransformerCLSRegressor(
#         in_dim=in_dim,
#         d_model=args.d_model,
#         nhead=args.n_heads,
#         num_layers=args.n_layers,
#         dim_feedforward=args.ffn_dim,
#         dropout=args.dropout,
#         cls_init=args.cls_init,
#         use_posenc=args.use_posenc,
#     ).to(dev)

#     # 6) Optim & loss
#     loss_fn = torch.nn.MSELoss(reduction="mean")
#     optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#     best_val = float("inf")
#     best_state = None
#     epoch_train_loss = []
#     epoch_valid_loss = []

#     for epoch in range(args.epochs):
#         model.train()
#         tl = 0.0
#         n = 0
#         for mem, mask, y, _ in tqdm(train_loader, desc=f"[Epoch {epoch}] train"):
#             y_hat = model(mem, mask)
#             loss = loss_fn(y_hat, y)
#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#             tl += loss.item() * y.size(0)
#             n += y.size(0)
#         tl /= max(n, 1)
#         epoch_train_loss.append(tl)

#         model.eval()
#         with torch.no_grad():
#             vl = 0.0
#             n = 0
#             for mem, mask, y, _ in tqdm(valid_loader, desc=f"[Epoch {epoch}] valid"):
#                 y_hat = model(mem, mask)
#                 loss = loss_fn(y_hat, y)
#                 vl += loss.item() * y.size(0)
#                 n += y.size(0)
#             vl /= max(n, 1)
#             epoch_valid_loss.append(vl)

#         print(f"Epoch {epoch}: train MSE={tl:.6f} | valid MSE={vl:.6f}")

#         # Early stop heuristic
#         if vl < best_val:
#             best_val = vl
#             best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

#         if (epoch > args.es_patience) and len(epoch_valid_loss) > (args.es_patience + 3):
#             recent = sum(epoch_valid_loss[-3:]) / 3
#             prev = sum(epoch_valid_loss[-(args.es_patience + 3):-3]) / args.es_patience
#             if recent > prev:
#                 print(f"Early stopping at epoch {epoch}")
#                 break

#     # Restore best
#     if best_state is not None:
#         model.load_state_dict(best_state)

#     # 7) Test (report in real meters)
#     y_std = ds_train.y_std.to(dev) if ds_train.y_std is not None else torch.tensor(1.0, device=dev)
#     y_mean = ds_train.y_mean.to(dev) if ds_train.y_mean is not None else torch.tensor(0.0, device=dev)

#     def eval_loader(loader):
#         model.eval()
#         se = 0.0
#         ae = 0.0
#         n = 0
#         with torch.no_grad():
#             for mem, mask, y, _ in loader:
#                 y_hat = model(mem, mask)
#                 # de-normalize
#                 y_hat_real = y_hat * y_std + y_mean
#                 y_real = y * y_std + y_mean
#                 se += torch.sum((y_hat_real - y_real) ** 2).item()
#                 ae += torch.sum(torch.abs(y_hat_real - y_real)).item()
#                 n += y.size(0)
#         mse = se / max(n, 1)
#         mae = ae / max(n, 1)
#         return mse, mae

#     test_mse, test_mae = eval_loader(test_loader)
#     print(f"TEST  MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")

#     return test_mse, test_mae
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Transformer_Map_Interp.datasets.transformerRegressorDataClass import (
    TransformerPointDataset, collate_point_batches
)
from Transformer_Map_Interp.datasets.dt2_data import load_dt2_data, load_multi_dt2_data
from Transformer_Map_Interp.models.transformerRegressor import TransformerCLSRegressor
from torch.utils.tensorboard import SummaryWriter

def debug_shape(phase, mem):
        B, S, F = mem.shape
        print(f"[{phase}] mem_tokens shape = (B={B}, S={S}, F={F})")

def run_transformer(args, tb_writer: SummaryWriter | None = None) -> Tuple[float, float]:
    """
    Trains and evaluates the decoder-only Transformer with a CLS token.
    Per-epoch logs: train/val loss (normalized), train/val MSE & MAE (in meters).
    Saves metrics to metrics.csv and metrics.pt in save_dir.
    """
    import os, csv

    # ---------------------------
    # helpers
    # ---------------------------
    def _ensure_dir(p: str):
        os.makedirs(p, exist_ok=True)

    def _save_metrics(save_dir: str, rows: list[dict]):
        _ensure_dir(save_dir)
        csv_path = os.path.join(save_dir, "metrics.csv")
        pt_path = os.path.join(save_dir, "metrics.pt")

        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
        torch.save(rows, pt_path)

    @torch.no_grad()
    def _eval_epoch(model, loader, y_mean: torch.Tensor, y_std: torch.Tensor, device: torch.device, phase) -> Dict[str, float]:
        """Returns dict with normalized loss, plus real-units MSE/MAE."""
        loss_fn = torch.nn.MSELoss(reduction="mean")
        model.eval()
        tot_loss = 0.0
        se_real = 0.0
        ae_real = 0.0
        n = 0
        for mem, mask, y, _ in loader:
            # (your dataset typically already yields tensors on device; .to is a no-op if so)
            mem, mask, y = mem.to(device), mask.to(device), y.to(device)
            #debug_shape(phase, mem)
            y_hat = model(mem, mask)
            loss = loss_fn(y_hat, y)
            tot_loss += loss.item() * y.size(0)

            # de-normalize for real metrics
            y_hat_real = y_hat * y_std + y_mean
            y_real = y * y_std + y_mean
            se_real += torch.sum((y_hat_real - y_real) ** 2).item()
            ae_real += torch.sum(torch.abs(y_hat_real - y_real)).item()
            n += y.size(0)

        return {
            "loss": tot_loss / max(n, 1),
            "mse": se_real / max(n, 1),
            "mae": ae_real / max(n, 1),
        }

    # ---------------------------
    # 1) Load data (train/val/test/calib) 
    # ---------------------------
    #trainset, validset, testset, calibset = load_dt2_data(args)  # calibset not used yet
    trainset, validset, testset, calibset = load_multi_dt2_data(args)  # calibset not used yet
    # ---------------------------
    # 2) Neighbor index over TRAIN only
    # ---------------------------
    # nei = NeighborIndex(train_coords=trainset.coords.cpu(), k=args.n_neighbors)

    # ---------------------------
    # 3) Point datasets (CLS-centered)
    # ---------------------------
    dev = args.device
    ds_train = TransformerPointDataset(trainset, trainset, device=dev)
    ds_valid = TransformerPointDataset(validset, trainset, device=dev)
    ds_test  = TransformerPointDataset(testset,  trainset, device=dev)

    # ---------------------------
    # 4) Dataloaders
    # ---------------------------
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_point_batches)
    valid_loader = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False, collate_fn=collate_point_batches)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_point_batches)

    # ---------------------------
    # 5) Model
    # ---------------------------
    in_dim = 3  # [Δlat, Δlon, y_known_norm, is_observed] # Orit 12.10 from 4 --> 3
    model = TransformerCLSRegressor(
        in_dim=in_dim,
        d_model=args.d_model,
        nhead=args.n_heads,       # keep your arg names as-is
        num_layers=args.n_layers, # keep your arg names as-is
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        #cls_init=args.cls_init,
        use_posenc=args.use_posenc,
    ).to(dev)

    # ---------------------------
    # 6) Optim & loss
    # ---------------------------
    loss_fn = torch.nn.MSELoss(reduction="mean")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # y stats for de-normalization
    y_std = ds_train.y_std if isinstance(ds_train.y_std, torch.Tensor) else torch.tensor(ds_train.y_std or 1.0)
    y_mean = ds_train.y_mean if isinstance(ds_train.y_mean, torch.Tensor) else torch.tensor(ds_train.y_mean or 0.0)
    y_std = y_std.to(dev)
    y_mean = y_mean.to(dev)

    # save location
    save_dir = getattr(args, "save_path",
                       f"saved_models/transformer_{getattr(args, 'dataset', 'dataset')}")

    best_val = float("inf")
    best_state = None
    metrics_rows: list[dict[str, Any]] = []

    # ---------------------------
    # Training epochs
    # ---------------------------
    for epoch in range(args.epochs):
        # ---- Train step (accumulate normalized train loss) ----
        model.train()
        train_loss_acc = 0.0
        n_train = 0 
        for mem, mask, y, _ in tqdm(train_loader, desc=f"[Epoch {epoch}] train"): #mem_tokens, pad_mask, y, cls_coords
            mem, mask, y = mem.to(dev), mask.to(dev), y.to(dev)
            #debug_shape("train", mem)
            y_hat = model(mem, mask)
            loss = loss_fn(y_hat, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss_acc += loss.item() * y.size(0)
            n_train += y.size(0)
        train_loss_norm = train_loss_acc / max(n_train, 1)

        # ---- Eval: train metrics (real units) ----
        train_eval = _eval_epoch(model, DataLoader(ds_train, batch_size=args.batch_size, shuffle=False,
                                                   collate_fn=collate_point_batches),
                                 y_mean, y_std, dev,"eval - train")

        # ---- Eval: val metrics ----
        val_eval = _eval_epoch(model, valid_loader, y_mean, y_std, dev, "eval - val")

        # ---- Log & keep best (by val normalized loss, same as your original logic) ----
        row = {
            "epoch": epoch,
            "train_loss_norm": train_loss_norm,
            "train_mse": train_eval["mse"],
            "train_mae": train_eval["mae"],
            "val_loss_norm": val_eval["loss"],
            "val_mse": val_eval["mse"],
            "val_mae": val_eval["mae"],
        }
        metrics_rows.append(row)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss_norm:.6f} | train_MSE={train_eval['mse']:.6f} | train_MAE={train_eval['mae']:.6f} || "
            f"val_loss={val_eval['loss']:.6f} | val_MSE={val_eval['mse']:.6f} | val_MAE={val_eval['mae']:.6f}"
        )
        
        if tb_writer:
            tb_writer.add_scalar("loss/train", train_loss_norm, epoch)
            tb_writer.add_scalar("loss/val",   val_eval['loss'] ,   epoch)

            tb_writer.add_scalar("mse_raw/train", train_eval['mse'], epoch)
            tb_writer.add_scalar("mse_raw/val",   val_eval['mse'],   epoch)
            tb_writer.add_scalar("mae_raw/train", train_eval['mae'], epoch)
            tb_writer.add_scalar("mae_raw/val",   val_eval['mae'],   epoch)

            tb_writer.add_scalar("lr", optim.param_groups[0]["lr"], epoch)

        if val_eval["loss"] < best_val:
            best_val = val_eval["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch > args.es_patience) and (len(metrics_rows) > (args.es_patience + 3)):
            recent = sum(r["val_loss_norm"] for r in metrics_rows[-3:]) / 3
            prev   = sum(r["val_loss_norm"] for r in metrics_rows[-(args.es_patience + 3):-3]) / args.es_patience
            if recent > prev:
                print(f"Early stopping at epoch {epoch}")
                break
    if tb_writer:
    # pack your args into a flat dict of strings/numbers
        hparams = {
            "d_model": args.d_model,
            "nhead": args.n_heads,
            "layers": args.n_layers,
            "ffn": args.ffn_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "dataset": str(args.dataset),
        }
    
    # ---------------------------
    # Restore best & save metrics/model
    # ---------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    _save_metrics(save_dir, metrics_rows)
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

    # ---------------------------
    # 7) Final test (real units)
    # ---------------------------
    @torch.no_grad()
    def _eval_loader(loader):
        r = _eval_epoch(model, loader, y_mean, y_std, dev, "eval - test")
        return r["mse"], r["mae"]

    test_mse, test_mae = _eval_loader(test_loader)
    print(f"TEST  MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")
        # final metrics you want shown in the hparams table:
    final_metrics = {
        "hp/test_mse_raw": float(test_mse),   # compute at test time
        "hp/test_mae_raw": float(test_mae),
    }
    if tb_writer:
        tb_writer.add_hparams(hparams, final_metrics)  
    return test_mse, test_mae
