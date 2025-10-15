# simple_metrics.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====== EDIT IF NEEDED ======
CSV  = "metrics.csv"     # path to your metrics CSV
SAVE = True             # True = save PNGs, False = show() interactively
OUT  = "metrics_plots"   # folder for saved figures if SAVE=True
# ============================

# Load the CSV (all columns become vectors you can access via df['col'])
df = pd.read_csv(CSV)

# Pick a horizontal axis (step/epoch). If none found, use the index.
STEP_CANDIDATES = ["step", "steps", "global_step", "epoch", "iter", "iteration", "batch"]
step_col = next((c for c in STEP_CANDIDATES if c in df.columns), None)
x = df[step_col].to_numpy() if step_col else np.arange(len(df))

# Detect loss/mse columns (case-insensitive)
loss_cols = [c for c in df.columns if "loss" in c.lower()]
mse_cols  = [c for c in df.columns if ("mse" in c.lower()) or ("mean_squared_error" in c.lower())]

# Fallbacks (in case your CSV uses different names)
if not loss_cols and "train_loss" in df.columns: loss_cols = ["train_loss"]
if not mse_cols and "train_mse" in df.columns:   mse_cols  = ["train_mse"]

# Print a tiny summary so you see what vectors were parsed
print(f"Step axis: {step_col if step_col else 'index'}  |  rows: {len(x)}")
print("Loss columns:", loss_cols or "(none found)")
print("MSE columns: ", mse_cols  or "(none found)")
for c in loss_cols + mse_cols:
    v = df[c].to_numpy()
    print(f"  {c}: len={len(v)}, last={v[-1] if len(v) else 'NA'}")

if SAVE:
    os.makedirs(OUT, exist_ok=True)

# ---- Plot LOSS ----
if loss_cols:
    plt.figure()
    for c in loss_cols:
        plt.plot(x, df[c].to_numpy(), label=c, linewidth=1.2)
    plt.xlabel(step_col if step_col else "index")
    plt.ylabel("loss")
    plt.title("Loss over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if SAVE:
        plt.savefig(os.path.join(OUT, "loss.png"), dpi=150)
        plt.close()
    else:
        plt.show()

# ---- Plot MSE ----
if mse_cols:
    plt.figure()
    for c in mse_cols:
        plt.plot(x, df[c].to_numpy(), label=c, linewidth=1.2)
    plt.xlabel(step_col if step_col else "index")
    plt.ylabel("MSE")
    plt.title("MSE over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if SAVE:
        plt.savefig(os.path.join(OUT, "mse.png"), dpi=150)
        plt.close()
    else:
        plt.show()
