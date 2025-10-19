# simple_y_hists.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== EDIT THESE IF NEEDED ====
CSV   = "y_values.csv"   # path to your csv
YCOL  = "y"              # the numeric column to plot (e.g., "elevation")
#SPLIT = ["split", "is_norm"]         # per-set column (set to None if you don't have one)
BINS  = 50               # histogram bins
SAVE  = True             # True=save PNGs, False=just show()
OUT   = "y_values_plots" # output folder if SAVE=True
# ==============================
df = pd.read_csv(CSV)

for (set_name, is_norm), sub in df.groupby(["split", "is_norm"]):
    y = sub["y"].to_numpy(dtype=float)
    if is_norm == 0:
        # raw y
        plt.figure()
        plt.hist(y[~np.isnan(y)], bins=BINS)
        plt.xlabel("y (raw)"); plt.ylabel("count")
        plt.title(f"y — set: {set_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"hist_y_{set_name}.png"), dpi=150)
        plt.show()
    else:
        # already-normalized y (DON'T re-normalize here)
        plt.figure()
        plt.hist(y[~np.isnan(y)], bins=BINS)
        plt.xlabel("y_norm"); plt.ylabel("count")
        plt.title(f"y_norm — set: {set_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"hist_y_norm_{set_name}.png"), dpi=150)
        plt.show()
# df = pd.read_csv(CSV)

# # choose groups (one set or many)
# if SPLIT and SPLIT in df.columns:
#     groups = df.groupby(SPLIT)
# else:
#     df["_set"] = "all"
#     groups = df.groupby("_set")

# if SAVE:
#     os.makedirs(OUT, exist_ok=True)

# for set_name, sub in groups:
#     # --- vectors you asked for ---
#     y = sub[YCOL].to_numpy(dtype=float)               # y
#     mu, sigma = y.mean(), y.std(ddof=0)
#     y_norm = (y - mu) / (sigma if sigma > 0 else 1.)  # y_norm (z-score)

#     # --- hist of y ---
#     plt.figure()
#     plt.hist(y[~np.isnan(y)], bins=BINS)
#     plt.xlabel(YCOL); plt.ylabel("count")
#     plt.title(f"{YCOL} — set: {set_name}")
#     plt.tight_layout()
#     if SAVE:
#         plt.savefig(os.path.join(OUT, f"hist_y_{set_name}.png"), dpi=150)
#         plt.close()
#     else:
#         plt.show()

#     # --- hist of y_norm ---
#     plt.figure()
#     plt.hist(y_norm[~np.isnan(y_norm)], bins=BINS)
#     plt.xlabel("y_norm (z-score)"); plt.ylabel("count")
#     plt.title(f"y_norm — set: {set_name}")
#     plt.tight_layout()
#     if SAVE:
#         plt.savefig(os.path.join(OUT, f"hist_y_norm_{set_name}.png"), dpi=150)
#         plt.close()
#     else:
#         plt.show()
