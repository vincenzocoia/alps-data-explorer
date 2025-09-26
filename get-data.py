# %%
# Import necessary libraries
import ee
import xarray as xr
import xrspatial as xs
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from streamlit_folium import st_folium
from scipy.stats import norm
import scipy.stats as stats
#st.set_page_config(layout="wide")
st.title('Alps Data Explorer')
col1, col2 = st.columns(2)

def kendall_tau_1d(x, y):
    """
    Calculate the Kendall tau correlation between two 1D arrays.
    - x: 1D array
    - y: 1D array
    - return: Kendall tau correlation
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    tau, _ = stats.kendalltau(x[mask], y[mask])
    return tau

# Helper: convert data to normal scores using rank-based uniformization
def _to_normal_scores(x, a=-0.5):
    """
    Convert data to normal scores using rank-based uniformization.
    - x: 1D array
    - a: parameter for the uniformization; -1 < a < 0 recommended.
    - return: 1D array of normal scores
    """
    x = np.asarray(x, dtype=float)
    mask = ~np.isnan(x)
    if mask.sum() == 0:
        return np.full_like(x, np.nan, dtype=float)
    # Rank non-NaN values (1..n), ties averaged to mirror R's default
    ranks = pd.Series(x[mask]).rank(method='average')
    n = int(mask.sum())
    # Uniform scores in (0,1) using (rank + a) / (n + 1 + 2a)
    u = (ranks + a) / (n + 1 + 2 * a)
    # Avoid exactly 0 or 1 to prevent +/-inf from ppf
    eps = np.finfo(float).eps
    u = np.clip(u.to_numpy(), eps, 1.0 - eps)
    z = norm.ppf(u)
    out = np.full_like(x, np.nan, dtype=float)
    out[mask] = z
    return out

# %%
# Initialize the Earth Engine module
ee.Authenticate()
ee.Initialize(project='alps-data-explorer')

# %%
# Alps polygon
import folium

# Create a map centered around the Alps
m = folium.Map(location=[44.5, 9.5], zoom_start=4)

# Quick polygon for the Alps
alps_polygon_coords = [
  [44.272, 8.274],
  [44.324, 7.494],
  [45.084, 7.305],
  [45.764, 8.852],
  [45.544, 10.796],
  [45.516, 15.044],
  [48.009, 14.821],
  [47.144, 8.596],
  [44.956, 5.120],
  [43.543, 6.744]
]

# Add the polygon to the map
folium.Polygon(
    alps_polygon_coords, color='blue', fill=True, fill_opacity=0.2
).add_to(m)

# Display the map
with col1:
    st_folium(m, width=700, height=500)

# %%
# Download some ERA5 data
@st.cache_data
def download_data(polygon_coords):
    ic = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate(
        '1992-10-05', '1993-03-31'
    )
    leg1 = ee.Geometry.Polygon(polygon_coords)
    ds = xr.open_dataset(
        ic,
        #engine='netcdf4',
        projection=ic.first().select(0).projection(),
        geometry=leg1
    )
    return ds

# Use the cached function to get the dataset
ds = download_data(alps_polygon_coords)

# %%
# Streamlit selectors
v = st.sidebar.selectbox("Select a variable", ds.data_vars)
v2 = st.sidebar.selectbox("Select aother variable", ds.data_vars)
choose_x = st.sidebar.select_slider("Select a x", ds.lon.values)
choose_y = st.sidebar.select_slider("Select a y", ds.lat.values)
agg_duration = st.sidebar.selectbox("Select an aggregation duration", ['monthly', 'yearly', 'daily'])
agg_func = st.sidebar.selectbox("Select an aggregation function", ['max', 'min', 'mean', 'sum'])
scale = st.sidebar.radio("Scale", ['Linear', 'Normal Scores'])
submit_button = st.sidebar.button(label='Refresh')

# %%
if submit_button:
    # Looking over time
    ds_xy = ds[[v, v2]].sel(lon=choose_x, lat=choose_y)
    
    # Aggregate the data
    if agg_duration == 'daily':
        ds_xy = ds_xy.resample(time='D').reduce(getattr(np, agg_func))
    elif agg_duration == 'monthly':
        ds_xy = ds_xy.resample(time='ME').reduce(getattr(np, agg_func))
    elif agg_duration == 'yearly':
        ds_xy = ds_xy.resample(time='Y').reduce(getattr(np, agg_func))
        
    # Extract the data for plotting
    time = ds_xy['time'].values
    v_data = ds_xy[v].values
    v2_data = ds_xy[v2].values
    
    # Apply normal scores only if requested
    if scale == 'Normal Scores':
        x_plot = _to_normal_scores(v_data)
        y_plot = _to_normal_scores(v2_data)
    else:
        x_plot = v_data
        y_plot = v2_data
    
    # Calculate kendall correlation for each pixel.
    da1 = ds[v]
    da2 = ds[v2]
    
    tau = xr.apply_ufunc(
        kendall_tau_1d,
        da1, da2,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True,
    ).rename('kendall_tau')
    
    with col2:
        plt.scatter(x_plot, y_plot)
        plt.xlabel(v)
        plt.ylabel(v2)
        plt.title(f'Scatter plot. Scale: {scale}')
        st.pyplot(plt.gcf())
    
    with col1:
        # Create a line plot for the first variable
        plt.figure(figsize=(7, 2))
        plt.plot(time, v_data, label=f'{v} over time')
        plt.xlabel('Time')
        plt.ylabel(v)
        plt.title(f'Line Plot of {v} over Time')
        plt.legend()
        st.pyplot(plt.gcf())
    
    with col2:
        # Create a line plot for the second variable
        plt.figure(figsize=(7, 2))
        plt.plot(time, v2_data, label=f'{v2} over time')
        plt.xlabel('Time')
        plt.ylabel(v2)
        plt.title(f'Line Plot of {v2} over Time')
        plt.legend()
        st.pyplot(plt.gcf())
