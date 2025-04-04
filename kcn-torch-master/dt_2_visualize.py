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
dt2_file = "datasets/n32_e035_1arc_v3.dt2"
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
