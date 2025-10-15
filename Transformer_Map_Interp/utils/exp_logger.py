# utils/exp_logger.py
import os, csv, json
from dataclasses import asdict, is_dataclass
import matplotlib.pyplot as plt

class ExpLogger:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.save_dir, "metrics.csv")
        self.rows = []

        # write header on first use
        self._header_written = False

    def _maybe_write_header(self, row_dict):
        if not self._header_written:
            with open(self.metrics_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
                w.writeheader()
                w.writerow(row_dict)
            self._header_written = True
        else:
            with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
                w.writerow(row_dict)

    def log_epoch(self, **kwargs):
        # kwargs: epoch, lr, train_loss, val_loss, train_mse_raw, train_mae_raw,
        #         val_mse_raw, val_mae_raw, train_mse_norm, train_mae_norm, val_mse_norm, val_mae_norm
        self.rows.append(kwargs)
        self._maybe_write_header(kwargs)

    def save_hyperparams_csv(self, args_or_dict):
        # flatten to a 1-row CSV
        d = {}
        if is_dataclass(args_or_dict):
            d = asdict(args_or_dict)
        elif hasattr(args_or_dict, "__dict__"):
            d = vars(args_or_dict).copy()
        elif isinstance(args_or_dict, dict):
            d = dict(args_or_dict)
        else:
            d = {"value": str(args_or_dict)}

        # also dump json for completeness
        with open(os.path.join(self.save_dir, "hyperparams.json"), "w", encoding="utf-8") as jf:
            json.dump(d, jf, indent=2, default=str)

        # CSV one-liner
        cols = sorted(d.keys())
        with open(os.path.join(self.save_dir, "hyperparams.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerow({k: d.get(k) for k in cols})

    def _plot(self, x, ys, labels, title, fname, ylabel=None):
        plt.figure()
        for y, lab in zip(ys, labels):
            plt.plot(x, y, label=lab)
        plt.title(title)
        if ylabel: plt.ylabel(ylabel)
        plt.xlabel("epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, fname), dpi=150)
        plt.close()

    def save_plots(self):
        if not self.rows:
            return
        # collect series
        epochs = [r["epoch"] for r in self.rows]
        t_loss = [r["train_loss"] for r in self.rows]
        v_loss = [r["val_loss"]   for r in self.rows]

        t_mse_raw = [r["train_mse_raw"] for r in self.rows]
        v_mse_raw = [r["val_mse_raw"]   for r in self.rows]
        t_mae_raw = [r["train_mae_raw"] for r in self.rows]
        v_mae_raw = [r["val_mae_raw"]   for r in self.rows]

        t_mse_n = [r["train_mse_norm"] for r in self.rows]
        v_mse_n = [r["val_mse_norm"]   for r in self.rows]
        t_mae_n = [r["train_mae_norm"] for r in self.rows]
        v_mae_n = [r["val_mae_norm"]   for r in self.rows]

        self._plot(epochs, [t_loss, v_loss], ["train", "val"], "Loss (as used for training)", "loss.png", "loss")
        self._plot(epochs, [t_mse_raw, v_mse_raw], ["train", "val"], "MSE (raw, meters^2)", "mse_raw.png", "MSE raw")
        self._plot(epochs, [t_mae_raw, v_mae_raw], ["train", "val"], "MAE (raw, meters)", "mae_raw.png", "MAE raw")
        self._plot(epochs, [t_mse_n, v_mse_n], ["train", "val"], "MSE (normalized)", "mse_norm.png", "MSE norm")
        self._plot(epochs, [t_mae_n, v_mae_n], ["train", "val"], "MAE (normalized)", "mae_norm.png", "MAE norm")
