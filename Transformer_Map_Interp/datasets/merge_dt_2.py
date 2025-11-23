import rasterio
from rasterio.merge import merge
import os
print(os.getcwd())

# === EDIT THESE PATHS ===
in1 = "./Transformer_Map_Interp/datasets/n32_e035_1arc_v3.dt2"          # first DTED file
in2 = "./Transformer_Map_Interp/datasets/n32_e034_1arc_v3.dt2"          # second DTED file
in3 = "./Transformer_Map_Interp/datasets/n33_e035_1arc_v3.dt2"          # second DTED file
out_merged = "./Transformer_Map_Interp/datasets/merged.tif"  # output file
# =========================

# Open the two rasters
src1 = rasterio.open(in1)
src2 = rasterio.open(in2)
src3 = rasterio.open(in3)
# Merge them
mosaic, out_transform = merge([src1,src3])

# Copy metadata from the first file
out_meta = src1.meta.copy()
out_meta.update({
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_transform,
    "driver": "GTiff",
})

# Save merged raster
with rasterio.open(out_merged, "w", **out_meta) as dst:
    dst.write(mosaic)

print("Saved merged raster to:", out_merged)
