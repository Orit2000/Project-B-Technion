from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from Transformer_Map_Interp.datasets.transformerRegressorDataClass import (
    TransformerPointDataset, collate_point_batches
)
from Transformer_Map_Interp.datasets.dt2_data import load_multi_dt2_data
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

    def _save_metrics(save_dir: str, rows: list[dict], Name: str = "regular"):
        _ensure_dir(save_dir)
        csv_path = os.path.join(save_dir, f"metrics_{Name}.csv")
        pt_path = os.path.join(save_dir, f"metrics_{Name}.pt")

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
        for mem, y, _ in loader:
            # (your dataset typically already yields tensors on device; .to is a no-op if so)
            #mem, mask, y = mem.to(device), mask.to(device), y.to(device)
            mem, y = mem.to(device), y.to(device)
            #debug_shape(phase, mem)
            y_hat = model(mem)
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
    trainset, validset, testset = load_multi_dt2_data(args)  # calibset not used yet
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
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(0.05 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
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
    batch_metrics_rows: list[dict[str, Any]] = []   # <--- NEW
    # ---------------------------
    # Training epochs
    # ---------------------------
    PATIENCE = 5  # check last 3 vs previous 3
    early_stop_triggered = False
    for epoch in range(args.epochs):
        # ---- Train step (accumulate normalized train loss) ----
        model.train()
        train_loss_acc = 0.0
        n_train = 0 
        if epoch == 0:
            batch_idx_max = 1000
        else:
            batch_idx_max = 3000
        for batch_idx, (mem, y, _) in enumerate( #(mem, mask, y, _) 
        tqdm(train_loader, desc=f"[Epoch {epoch}] train"), start=1): #mem_tokens, pad_mask, y, cls_coords
            #mem, mask, y = mem.to(dev), mask.to(dev), y.to(dev)
            mem, y = mem.to(dev),  y.to(dev)
            #print(f"Mask: {mask}")
            optim.zero_grad()
            #debug_shape("train", mem)
            #y_hat = model(mem, mask)
            y_hat = model(mem)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optim.step()
            scheduler.step()
            train_loss_acc += loss.item() * y.size(0)
            n_train += y.size(0)
            # --- log every 1000 batches ---
            if (batch_idx % batch_idx_max) == 0:
                #train_loss_norm = train_loss_acc / max(n_train, 1)
                val_eval = _eval_epoch(model, valid_loader, y_mean, y_std, dev, "eval - val")
                y_hat_real = y_hat * y_std + y_mean
                y_real     = y * y_std + y_mean
                batch_row = {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "train_loss_norm": loss.item(),
                    "train_loss": float(torch.mean((y_hat_real- y_real)**2).item()),  # approximate real units for batch
                    "train_mae": float(torch.mean(torch.abs(y_hat_real - y_real)).item()),
                    "val_loss_norm": val_eval["loss"],
                    "val_mse": val_eval["mse"],
                    "val_mae": val_eval["mae"],
                    "lr": optim.param_groups[0]["lr"],
                }
                
                print(
                    f"Batch {batch_idx}: "
                    f"train_loss={loss.item():.6f} | train_MSE={batch_row['train_loss']:.6f} | train_MAE={batch_row['train_mae']:.6f} || "
                    f"val_loss={val_eval['loss']:.6f} | val_MSE={val_eval['mse']:.6f} | val_MAE={val_eval['mae']:.6f}"
                )
                batch_metrics_rows.append(batch_row)
                
                if len(batch_metrics_rows) >= 2 * PATIENCE:
                    # last 3 validation losses
                    recent_vals = [r["val_loss_norm"] for r in batch_metrics_rows[-PATIENCE:]]

                    # previous 3 validation losses
                    prev_vals = [r["val_loss_norm"] for r in batch_metrics_rows[-2*PATIENCE:-PATIENCE]]

                    if (sum(recent_vals) / PATIENCE) >= (sum(prev_vals) / PATIENCE):
                        print("\n🔥 Early stopping inside epoch: validation not improving.")
                        print(f"Stopped at batch {batch_idx} of epoch {epoch}.")
                        
                        # Save checkpoint before breaking
                        ckpt_path = os.path.join(save_dir, f"stopped_early_at_epoch_{epoch}_batch{batch_idx}.pt")
                        torch.save({
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "model_state": model.state_dict(),
                            "optimizer_state": optim.state_dict(),
                            "metrics_epoch": metrics_rows,
                            "metrics_batch": batch_metrics_rows,
                        }, ckpt_path)
                        print(f"Checkpoint saved to: {ckpt_path}\n")
                        early_stop_triggered = True
                        # break out of the batch loop (finish epoch)
                        break
                    
                if val_eval["loss"] < best_val:
                    best_val = val_eval["loss"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_{epoch}_batch_{batch_idx}.pt"))
                    print(f" New best model at epoch {epoch}, batch {batch_idx} with val_loss={best_val:.6f}")
                    _save_metrics(save_dir, batch_metrics_rows,"batch")
                        
                if tb_writer:
                    global_step = epoch * len(train_loader) + batch_idx
                    tb_writer.add_scalar("loss/train_batch", loss.item(), global_step)  
                                       
        if early_stop_triggered:
            break    
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

        # if val_eval["loss"] < best_val:
        #     best_val = val_eval["loss"]
        #     best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        #     torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_{epoch}.pt"))
        #     print(f" New best model at epoch {epoch} with val_loss={best_val:.6f}")
            
        # if (epoch % 1) == 0:
        #     ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
        #     torch.save({
        #         "epoch": epoch + 1,
        #         "model_state": model.state_dict(),
        #         "optimizer_state": optim.state_dict(),
        #         "best_val": best_val,
        #         "metrics_epoch": metrics_rows,
        #         "metrics_batch": batch_metrics_rows,
        #         "hparams": vars(args),   # <--- add this line
        #     }, ckpt_path)
        #     print(f" 💾 Checkpoint saved at epoch {epoch+1} -> {ckpt_path}")
        #     if tb_writer:
        #         tb_writer.add_text("checkpoint", f"Saved checkpoint (epoch {epoch+1})", epoch)

        # if (epoch > args.es_patience) and (len(metrics_rows) > (args.es_patience + 3)):
        #     recent = sum(r["val_loss_norm"] for r in metrics_rows[-3:]) / 3
        #     prev   = sum(r["val_loss_norm"] for r in metrics_rows[-(args.es_patience + 3):-3]) / args.es_patience
        #     if recent > prev:
        #         print(f"Early stopping at epoch {epoch}")
        #         break
        
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
    # if best_state is not None:
    #     model.load_state_dict(best_state)

    _save_metrics(save_dir, metrics_rows)
    _save_metrics(save_dir, batch_metrics_rows,"batch")
    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_model.pt"))

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
