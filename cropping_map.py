import rasterio
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
import numpy as np

def degrees_to_meters(degree_res, latitude):
    """Convert resolution from degrees to meters."""
    # 1 degree latitude ≈ 111,320 meters
    lat_res_m = degree_res * 111320  
    
    # 1 degree longitude ≈ 111,320 * cos(latitude) meters
    lon_res_m = degree_res * 111320 * np.cos(np.radians(latitude))  
    
    return lat_res_m, lon_res_m  

def cropped_to_tiff(dt2_file, output_path):
    # Open and crop
    with rasterio.open(dt2_file) as src:
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
        cropped = src.read(1, window=window)
        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update({
            "height": cropped.shape[0],
            "width": cropped.shape[1],
            "transform": transform,
            "driver": "GTiff"
        })

    # Save as GeoTIFF
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cropped, 1)

# -------------------- Parameters --------------------
dt2_file = "datasets/n32_e035_1arc_v3.dt2"
min_lon, max_lon = 35.03, 35.2  # Avoid the sea!
min_lat, max_lat = 32.8, 33.0

# -------------------- Load Full Map --------------------
with rasterio.open(dt2_file) as src:
    elevation = src.read(1)
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    degree_res_x, degree_res_y  = src.res  # Pixel size (degrees per pixel)
    bounds = src.bounds  # Geographic extent (min/max lon, lat)
    mid_latitude = (src.bounds.top + src.bounds.bottom) / 2

    # Convert resolution to meters
    lat_res_m, lon_res_m = degrees_to_meters(degree_res_x, mid_latitude)
vmin = np.min(elevation)
vmax = np.max(elevation)
# -------------------- Plot Full Map with Crop Box --------------------
plt.figure(figsize=(10, 8))
plt.imshow(elevation, cmap='terrain', extent=extent, origin='upper', vmin=vmin, vmax=vmax)
plt.colorbar(label='Elevation (m)')
plt.title("Full DT2 Map with Crop Box")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Draw red bounding box
plt.plot([min_lon, min_lon, max_lon, max_lon, min_lon],
         [min_lat, max_lat, max_lat, min_lat, min_lat],
         'r-', linewidth=2)

plt.grid(True)
plt.show()

# Prints
print(f"Resolution in Degrees: {degree_res_x}° x {degree_res_y}°")
print(f"Resolution in Meters: {lat_res_m:.2f}m x {lon_res_m:.2f}m")
#print(f"Size: {width} x {height} pixels")
print(f"Extent: {bounds}")
print(f"ranges of elevations: [{int(np.min(elevation)), int(np.max(elevation))}]")

# -------------------- Crop Operation --------------------
with rasterio.open(dt2_file) as src:
    window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
    cropped = src.read(1, window=window)
    transform = src.window_transform(window)

# ------------------- Saving cropped ---------------------
# output_path = "datasets/n32_e035_1arc_v3_cropped_tiff.tiff"
# with rasterio.open(dt2_file) as src:
#     window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
#     cropped = src.read(1, window=window)
#     transform = src.window_transform(window)

#     profile = src.profile.copy()
#     profile.update({
#         "height": cropped.shape[0],
#         "width": cropped.shape[1],
#         "transform": transform,
#         "tiled": False,  # <--- Add this line to avoid block-size constraint
#         "driver": "GTiff"
#     })

#     # Save as GeoTIFF
#     with rasterio.open(output_path, "w", **profile) as dst:
#         dst.write(cropped, 1)

# Calculate extent of cropped image
top_left = transform * (0, 0)
bottom_right = transform * (cropped.shape[1], cropped.shape[0])
extent_crop = [top_left[0], bottom_right[0], bottom_right[1], top_left[1]]

# -------------------- Plot Cropped Area --------------------
plt.figure(figsize=(8, 6))
plt.imshow(cropped, cmap='terrain', extent=extent_crop, origin='upper',  vmin=vmin, vmax=vmax)
plt.colorbar(label='Elevation (m)')
plt.title("Cropped DT2 Region")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

print(f"ranges of elevations_cropped: [{int(np.min(cropped)), int(np.max(cropped))}]")

