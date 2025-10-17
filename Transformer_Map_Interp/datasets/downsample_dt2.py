import numpy as np
import rasterio
from affine import Affine

# =============================
# USER PARAMETERS
# =============================
dt2_file = "Transformer_Map_Interp/datasets/n32_e034_1arc_v3.dt2"
out_tif = "Transformer_Map_Interp/datasets/n32_e034_1arc_v3_sampled.tif"

radius_km = 0.75        # search radius
k_neighbors = 100        # desired points inside that radius
safety = 1.0             # >1 => fewer points (coarser)
random_seed = 42
datasampling = "uniform" # or "none" to keep all
# =============================


# ---- Helper functions ----
def meters_per_deg(lat_deg):
    lat_m = 111_320.0
    lon_m = 111_320.0 * np.cos(np.radians(lat_deg))
    return lat_m, lon_m

def target_m_from_neighbors(radius_km, k, safety=1.0):
    """Compute smart target spacing from neighbor goal."""
    R = radius_km * 1000.0
    s = R * np.sqrt(np.pi / float(max(k, 1)))
    return float(s * safety)

def current_cell_meters(bounds, res):
    """Return current meters/pixel (xres_m, yres_m)."""
    xdeg, ydeg = float(res[0]), abs(float(res[1]))
    mid_lat = (bounds.top + bounds.bottom) / 2.0
    lat_m, lon_m = meters_per_deg(mid_lat)
    return lon_m * xdeg, lat_m * ydeg

def keep_fraction(xres_m, yres_m, target_m):
    """Fraction of points to keep."""
    cur_area = xres_m * yres_m
    des_area = target_m ** 2
    return np.clip(cur_area / des_area, 0.0, 1.0)


# ---- Load original DTED ----
with rasterio.open(dt2_file) as src:
    elevation = src.read(1)
    profile = src.profile.copy()
    bounds = src.bounds
    xres_m, yres_m = current_cell_meters(bounds, src.res)
    nodata = src.nodata if src.nodata is not None else -32767

# ---- Compute smart sampling fraction ----
target_m = target_m_from_neighbors(radius_km, k_neighbors, safety)
keep_frac = keep_fraction(xres_m, yres_m, target_m)

print(f"Current resolution:  x={xres_m:.2f} m  y={yres_m:.2f} m")
print(f"Target spacing:      {target_m:.1f} m  -> keep ≈ {keep_frac*100:.3f}%")

# ---- Flatten elevation map to point list ----
H, W = elevation.shape
lon = np.linspace(bounds.left, bounds.right, W)
lat = np.linspace(bounds.bottom, bounds.top, H)
lon_grid, lat_grid = np.meshgrid(lon, lat)
coords = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=1)
values = elevation.ravel()

total = coords.shape[0]
keep_n = int(total * keep_frac)
rng = np.random.RandomState(random_seed)
selected_idx = rng.choice(total, size=keep_n, replace=False) if datasampling == 'uniform' else np.arange(total)

print(f"Selected {keep_n:,}/{total:,} points ({keep_frac*100:.3f}%)")

# ---- Create mask of kept points ----
mask = np.zeros(total, dtype=bool)
mask[selected_idx] = True
mask_2d = mask.reshape(H, W)

# Option A: Keep only selected points (others as nodata)
downsampled = np.where(mask_2d, elevation, nodata).astype(np.float32)

# ---- Save to GeoTIFF ----
profile.update(dtype='float32', compress='lzw', nodata=nodata)
with rasterio.open(out_tif, "w", **profile) as dst:
    dst.write(downsampled, 1)

print(f"Saved downsampled map to: {out_tif}")
