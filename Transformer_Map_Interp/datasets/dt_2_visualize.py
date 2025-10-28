import rasterio
import numpy as np
import matplotlib.pyplot as plt

def degrees_to_meters(degree_res, latitude):
    """Convert resolution from degrees to meters."""
    # 1 degree latitude ≈ 111,320 meters
    lat_res_m = degree_res * 111320  
    
    # 1 degree longitude ≈ 111,320 * cos(latitude) meters
    lon_res_m = degree_res * 111320 * np.cos(np.radians(latitude))  
    
    return lat_res_m, lon_res_m 

# Load the DTED file
dt2_file = "Transformer_Map_Interp/datasets/n32_e035_1arc_v3_cropped_test_.tiff"

with rasterio.open(dt2_file) as dataset:
    elevation = dataset.read(1)  # Read the first band (elevation values)
    extent = [dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top]  # Get geographical extent
    width = dataset.width   # Number of columns (longitude points)
    height = dataset.height  # Number of rows (latitude points)
    degree_res_x, degree_res_y  = dataset.res  # Pixel size (degrees per pixel)
    bounds = dataset.bounds  # Geographic extent (min/max lon, lat)
    # Compute midpoint latitude for accurate conversion
    mid_latitude = (dataset.bounds.top + dataset.bounds.bottom) / 2

    # Convert resolution to meters
    lat_res_m, lon_res_m = degrees_to_meters(degree_res_x, mid_latitude)
    

# Prints
print(f"Resolution in Degrees: {degree_res_x}° x {degree_res_y}°")
print(f"Resolution in Meters: {lat_res_m:.2f}m x {lon_res_m:.2f}m")
print(f"Size: {width} x {height} pixels")
print(f"Extent: {bounds}")

# Plot the elevation data
plt.figure(figsize=(10, 8))
plt.imshow(elevation, cmap="terrain", extent=extent, origin="upper")
plt.colorbar(label="Elevation (m)")
plt.title("DTED Level 2 Elevation Data")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
plt.savefig("Transformer_Map_Interp/datasets/dt2_region.png")

# --- Elevation histogram ---
# Flatten and drop masked (NoData) values
elev_vals = elevation.flatten()  # same as np.array(elevation[~elevation.mask])

# Optional: clip extreme outliers for nicer visualization (comment out if not desired)
# q1, q99 = np.percentile(elev_vals, [1, 99])
# elev_vals = elev_vals[(elev_vals >= q1) & (elev_vals <= q99)]

plt.figure(figsize=(9, 6))
plt.hist(elev_vals, bins=100)  # adjust bins as you like
plt.xlabel("Elevation (m)")
plt.ylabel("Count")
plt.title("Elevation Histogram")
plt.tight_layout()
plt.savefig("Transformer_Map_Interp/datasets/dt2_elevation_hist.png", dpi=200)
plt.show()