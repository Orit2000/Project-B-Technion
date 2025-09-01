import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from numpy.polynomial import Polynomial
import rasterio

# Different Options:
# Option A: Constant-latitude line (a row)
def sample_cons_lat(dt2_map,lon_grid, lat_grid, lat_value):
    lat_idx = np.abs(lat_grid - lat_value).argmin()
    values_sampled = dt2_map[lat_idx, :]
    distance = np.arange(len(lon_grid))  # or compute distance if needed

    return distance, values_sampled

# Option B: Constant-longtitude line (a column)
#def sample_cons_lon(dt2_map,lon_idx): ## TODO: fix this
    #lon_grid = dt2_map[:] ## Complete
    #lat_grid = # Complete
    #lon_value = lon_grid[lon_idx]
    #values_sampled = dt2_map[:, lon_idx]
    #distance = np.arange(len(lat_grid))

    #return distance, values_sampled

# Load the DTED file
dt2_file = "datasets/n32_e035_1arc_v3.dt2"
with rasterio.open(dt2_file) as src:
    dt2_map = src.read(1)  # elevation values
    transform = src.transform
    width = src.width
    height = src.height

    # Create latitude and longitude arrays
    lon_grid = np.array([transform[2] + i * transform[0] for i in range(width)])
    lat_grid = np.array([transform[5] + j * transform[4] for j in range(height)])

lat_value = 32.3
distance, values_sampled = sample_cons_lat(dt2_map,lat_grid, lon_grid,lat_value)

# Fit polynomial
degree = 15
coeffs = np.polyfit(distance, values_sampled, degree)
poly = np.poly1d(coeffs)

# Evaluate on dense points if needed
dense_x = np.linspace(distance[0], distance[-1], 500)
fitted_values = poly(dense_x)

plt.figure(figsize=(10, 5))
plt.plot(distance, values_sampled, 'bo', label='Sampled values (grid)')
plt.plot(dense_x, fitted_values, 'r-', label=f'{degree}° Polynomial fit')
plt.xlabel("Index along line (or distance)")
plt.ylabel("dt2 value")
plt.title("Polynomial fit to grid-sampled values")
plt.grid(True)
plt.legend()
plt.show()
