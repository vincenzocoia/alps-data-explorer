# Interactive Exploration of core functions

# %%
# Import core functions
import core
from config import DEFAULT_CONFIG
import matplotlib.pyplot as plt

DEFAULT_CONFIG

# %%
# Initialize Earth Engine
core.initialize_earth_engine(DEFAULT_CONFIG['ee_project'])

# %%
# Plot Alps polygon
alps = DEFAULT_CONFIG['polygon_coords']
core.plot_alps_polygon(alps)

# %%
# Set date range
date_range = DEFAULT_CONFIG['date_range']
date_range = ("1993-01-01", "1995-12-31")
date_range

# %%
# Download ERA5 data
ds = core.download_era5_data(
  alps, 
  date_range=date_range,
  agg_duration="annual",
  agg_func="max",
  variables=["total_precipitation_hourly", "surface_runoff_hourly"]
)
print(f"✅ Data downloaded: {list(ds.data_vars)}")
print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
print(f"Spatial extent: {ds.lon.values.min():.2f} to {ds.lon.values.max():.2f} lon, {ds.lat.values.min():.2f} to {ds.lat.values.max():.2f} lat")

# %%
# Explore data variables
ds.data_vars

# %%
# Select variables of interest
v1 = 'total_precipitation_hourly'
v2 = 'surface_runoff_hourly'

# %%
ds_v1v2 = core.pluck_v1v2(ds, v1, v2)

# %%
ds_m1m2 = core.xr_transform_aggregate(ds_v1v2, 'yearly', 'max')

# %%
# Check that the annual max worked. Make time series comparing the two.
core.biv_timeseries_plots(ds_v1v2)
core.biv_timeseries_plots(ds_m1m2)

# %%
# Normal scores scatterplot
x = 11.0
y = 46.0
ds_m1m2_xy = core.pluck_xy(ds_m1m2, x, y)
core.biv_dependence_scatterplot(ds_m1m2_xy)

# %%
# Kendall's tau heatmap, with dot where scatterplot is.
tau = core.xr_transform_ktau(ds_m1m2)
fig, ax = core.xr_heatmap(tau)
plt.scatter(x, y)
plt.show()
