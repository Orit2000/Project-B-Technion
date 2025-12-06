import rasterio
import numpy as np
import matplotlib.pyplot as plt

def meters_per_degree(latitude_deg):
    lat_m = 111_320.0  # meters per 1 degree latitude (good enough for your scale)
    lon_m = 111_320.0 * np.cos(np.radians(latitude_deg))
    return lat_m, lon_m

# Load the DTED file
dt2_file = "Transformer_Map_Interp/datasets/n32_e035_1arc_v3.dt2" 
#dt2_file = "Transformer_Map_Interp/datasets/n33_e035_1arc_v3.dt2"
with rasterio.open(dt2_file) as dataset:
    #nodata = dataset.nodata
    bounds = dataset.bounds
    width, height = dataset.width, dataset.height
    xres_deg, yres_deg = dataset.res  # deg/pixel (yres will likely be negative)
    xres_deg = float(xres_deg)
    yres_deg = abs(float(yres_deg))
     # FIX: read the elevation correctly
    elevation = dataset.read(1).astype(float)

    #elevation = np.ma.masked_equal(elevation, nodata)

    # Mid-latitude for better lon->meters conversion
    mid_lat = (bounds.top + bounds.bottom) / 2.0

    # meters per degree at this latitude
    lat_m_per_deg, lon_m_per_deg = meters_per_degree(mid_lat)

    # meters per pixel
    xres_m = lon_m_per_deg * xres_deg
    yres_m = lat_m_per_deg * yres_deg
    transform = dataset.transform # affine transform for pixel->geo coords
    # total map size in meters
    width_m  = xres_m * width
    height_m = yres_m * height

    # Also keep extent for plotting
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

print(f"Resolution (deg/pixel): {xres_deg:.8f}° x {yres_deg:.8f}°")
print(f"Resolution (meters/pixel): {xres_m:.2f} m x {yres_m:.2f} m")
print(f"Map size: {width} x {height} pixels")
print(f"Map size: {width_m/1000:.3f} km (W) × {height_m/1000:.3f} km (H)")
print(f"Extent (lon/lat): {bounds}")

#valid = ~elevation.mask                     # boolean mask of kept pixels
#vals = elevation#[valid]                     # elevations at those pixels (1D)
#rows, cols = np.where(valid)
# pixel indices -> geographic coords (lon, lat)
#xs, ys = rasterio.transform.xy(transform, rows, cols)
# Plot (save before show to ensure file is written)
plt.figure(figsize=(10, 8))
plt.imshow(elevation, cmap="terrain", extent=extent, origin="upper")
#plt.scatter(xs, ys, c=vals, s=2, cmap="terrain", marker='.')
plt.colorbar(label="Elevation (m)")
plt.title("DTED Level 2 Elevation Data")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig("Transformer_Map_Interp/datasets/full_dt2_region.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 8))
plt.hist(elevation, bins=50)  # adjust bins as you like
plt.xlabel("Elevation (m)")
plt.ylabel("Count")
plt.title("Elevation Histogram")
plt.tight_layout()
plt.savefig("Transformer_Map_Interp/datasets/full_dt2_elevation_hist.png", dpi=200)
plt.show()

print("Elevation dtype:", elevation.dtype)
# print("Nodata value:", nodata)
# print("Masked values count:", np.sum(elevation.mask))
# print("Valid values count:", np.sum(~elevation.mask))
print("Min/Max of valid data:", elevation.min(), elevation.max())
num_points = total_points = elevation.size
total_points = elevation.size
print(f"Number of valid (non-nodata) points: {num_points:,}")
print(f"Fraction of valid points: {100 * num_points / total_points:.3f}%")