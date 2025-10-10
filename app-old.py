# %%
# Import necessary libraries
import ee
import xarray as xr
#import xrspatial as xs
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
from streamlit_folium import st_folium
import scipy.stats as stats
from utils import nscore

# Detect if running interactively (Jupyter/IPython) vs Streamlit web app
try:
    # Check if we're in Jupyter/IPython
    get_ipython()
    interactive = True
except NameError:
    # Not in Jupyter, assume Streamlit web app
    interactive = False
#st.set_page_config(layout="wide")
st.title('Alps Data Explorer')
col1, col2 = st.columns(2)


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
def download_data(polygon_coords, date_range=('1992-10-05', '1993-03-31'), 
                  agg_duration='none', agg_func='mean', variables=None):
    """
    Download ERA5 data with optional aggregation using xee
    
    Parameters
    ----------
    polygon_coords : list
        List of coordinate pairs defining the polygon
    date_range : tuple, optional
        Start and end dates for data collection, by default ('1992-10-05', '1993-03-31')
    agg_duration : str, optional
        Aggregation duration ('daily', 'monthly', 'yearly', 'none'), by default 'none'
    agg_func : str, optional
        Aggregation function ('mean', 'max', 'min', 'sum'), by default 'mean'
    variables : list, optional
        List of specific variables to download, by default None (all variables)
    """
    from core import download_era5_data
    return download_era5_data(polygon_coords, date_range, agg_duration, agg_func, variables)

# Use the cached function to get the dataset
ds = download_data(alps_polygon_coords, 
                   date_range=('1992-10-05', '1993-03-31'),
                   agg_duration='none', 
                   agg_func='mean')

# %%
# Streamlit selectors
# If interactive, we can use the following selectors, else use defaults.


if interactive:
    v = 'snowmelt'
    v2 = 'surface_runoff_hourly'
    choose_x = ds.lon.values[0]
    choose_y = ds.lat.values[0]
    agg_duration = 'monthly'
    agg_func = 'mean'
    scale = 'Linear'
    submit_button = True
else:
    v = st.sidebar.selectbox("Select a variable", ds.data_vars)
    v2 = st.sidebar.selectbox("Select another variable", ds.data_vars)
    choose_x = st.sidebar.select_slider("Select a x", ds.lon.values)
    choose_y = st.sidebar.select_slider("Select a y", ds.lat.values)
    agg_duration = st.sidebar.selectbox(
        "Select an aggregation duration", 
        ['monthly', 'yearly', 'daily']
    )
    agg_func = st.sidebar.selectbox(
        "Select an aggregation function",
        ['max', 'min', 'mean', 'sum']
    )
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
        x_plot = nscore(v_data)
        y_plot = nscore(v2_data)
    else:
        x_plot = v_data
        y_plot = v2_data
    
    # Calculate kendall correlation for each pixel.
    da1 = ds[v]
    da2 = ds[v2]
    
    # Use scipy.stats.kendalltau directly with xarray
    # We need a wrapper to return only the tau value (not p-value)
    def kendall_tau_only(x, y):
        tau, _ = stats.kendalltau(x, y)
        return tau
    
    tau = xr.apply_ufunc(
        kendall_tau_only,
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
