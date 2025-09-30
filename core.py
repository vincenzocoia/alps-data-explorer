"""
Core analysis functions for Alps Data Explorer
Separates business logic from UI concerns
"""

# %%
import ee
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from utils import nscore
from config import DEFAULT_CONFIG
import folium
from typing import Tuple, Dict, Any

# %%
def initialize_earth_engine(project: str = None):
    """
    Initialize Earth Engine with authentication
    
    Parameters
    ----------
    project : str, optional
        The Earth Engine project to use. Defaults to the project specified in the DEFAULT_CONFIG.
        
    Returns
    -------
    None
    """
    project = project or DEFAULT_CONFIG['ee_project']
    ee.Authenticate()
    ee.Initialize(project=project)

# %%
def plot_alps_polygon(polygon_coords: list) -> folium.Map:
    """Create folium map with given polygon coordinates"""
    m = folium.Map(location=[44.5, 9.5], zoom_start=4)
    folium.Polygon(
        polygon_coords, color='blue', fill=True, fill_opacity=0.2
    ).add_to(m)
    return m

# %%
def download_era5_data(polygon_coords: list, date_range: Tuple[str, str]) -> xr.Dataset:
    """
    Download ERA5 data for given polygon and date range
    """
    
    ic = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate(*date_range)
    leg1 = ee.Geometry.Polygon(polygon_coords)
    
    ds = xr.open_dataset(
        ic,
        projection=ic.first().select(0).projection(),
        geometry=leg1
    )
    # Indicate that no aggregation has been applied.
    ds['aggregation'] = 'none'
    return ds

# %%
def pluck_two_variables(ds: xr.Dataset, 
                        v1: str, 
                        v2: str, 
                        agg_duration: str = 'monthly', 
                        agg_func: str = 'mean') -> xr.Dataset:
    """
    Zero-in on two variables of interest in the xarray dataset, with an 
    aggregation function applied for a specified duration.
    """
    ds = ds[[v1, v2]]
    if agg_duration == 'daily':
        ds = ds.resample(time='D').reduce(getattr(np, agg_func))
    elif agg_duration == 'monthly':
        ds = ds.resample(time='ME').reduce(getattr(np, agg_func))
    elif agg_duration == 'yearly':
        ds = ds.resample(time='Y').reduce(getattr(np, agg_func))
    # Add aggregation to the xarray dictionary.
    ds['aggregation'] = f'{agg_duration}_{agg_func}'
    return ds

# %%
def data_at_xy(ds: xr.Dataset,
               choose_x: float,
               choose_y: float) -> Dict[str, Any]:
    """
    Parse out the numpy arrays of the xarray variables over time, at a specific
    point. Also return normal scores of the variables.
    
    Returns dictionary with:
    - time: time values
    - arrays: numpy arrays of the variables of interest
    - nscores: numpy arrays of the normal scores of the variables of interest
    - location: (choose_x, choose_y)
    - aggregation: string describing the aggregation (duration and function).
    """
    # Extract time series at specific location
    ds_xy = ds.sel(lon=choose_x, lat=choose_y, method='nearest')

    # Extract data arrays
    time = ds_xy['time'].values
    variables = ds_xy.variables
    arrays = {}
    nscores = {}
    for v in variables:
        arrays[v] = ds_xy[v].values
        nscores[v] = nscore(arrays[v])
    return {
        'time': time,
        'arrays': arrays,
        'nscores': nscores,
        'location': (choose_x, choose_y),
        'aggregation': ds['aggregation']
    }

# %%
def timeseries_plot(time: np.ndarray, y: np.ndarray, yname: str) -> plt.Figure:
    """Create matplotlib timeseries plot"""
    fig_ts = plt.figure(figsize=(10, 4))
    plt.plot(time, y, label=f'{yname} over time')
    plt.xlabel('Time')
    plt.ylabel(yname)
    plt.title(f'Line Plot of {yname} over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig_ts

# %%
def biv_timeseries_plots(biv_xarray: Dict[str, Any]) -> Dict[str, plt.Figure]:
    """
    Create matplotlib time series plots of variables of interest from 
    the bivariate dependence results, `biv_dependence_data()`.
    
    Inputs:
    - biv_dependence_data: dictionary of bivariate dependence results from
      `biv_dependence_data()` function.
    
    Returns:
    - dictionary of two time series plots, one for each variable of interest.
    """
    assert len(biv_xarray.variables) == 2, "Bivariate xarray must have two variables."
    v1, v2 = biv_xarray.variables
    time = biv_xarray['time']
    v1_data = biv_xarray[v1].values
    v2_data = biv_xarray[v2].values
    fig_ts1 = timeseries_plot(time, v1_data, v1)
    fig_ts2 = timeseries_plot(time, v2_data, v2)
    plots = {
        'timeseries1': fig_ts1,
        'timeseries2': fig_ts2
    }
    return plots

# %%
def biv_dependence_scatterplot(biv_xarray: Dict[str, Any], 
                               nscore: bool = False) -> Dict[str, plt.Figure]:
    """
    Create matplotlib scatter plot of two variables of interest from
    the bivariate dependence results, `biv_dependence_data()`.
    
    Inputs:
    - biv_dependence_data: dictionary of bivariate dependence results from
      `biv_dependence_data()` function.
    - nscore: boolean indicating whether to use normal scores for the scatter 
      plot.
    
    Returns:
    - scatter plot of two variables of interest.
    """
    # Grab the data from the bivariate dependence processed data.
    assert len(biv_xarray.variables) == 2, "Bivariate xarray must have two variables."
    v1, v2 = biv_xarray.variables
    v1_data = biv_xarray[v1].values
    v2_data = biv_xarray[v2].values
    rho = stats.kendalltau(v1_data, v2_data).correlation
    scale = 'Linear'
    if nscore:
        v1_data = nscore(v1_data)
        v2_data = nscore(v2_data)
        scale = 'Normal Scores'

    # Create the scatter plot.
    fig_scatter = plt.figure(figsize=(8, 6))
    plt.scatter(v1_data, v2_data)
    plt.xlabel(v1)
    plt.ylabel(v2)
    plt.title(f'Scatter plot. Scale: {scale}. Kendall\'s tau: {rho:.2f}')
    
    return fig_scatter

# %%
def biv_dependence_correlation_map(ds: Dict[str, Any], v1: str, v2: str) -> xr.Dataset:
    """
    Create xarray dataset of Kendall's tau correlations for each (x, y)
    location for two variables of interest.
    
    Inputs:
    - ds: xarray dataset of two variables of interest.
    
    Returns:
    - xarray dataset of Kendall's tau correlations for each (x, y)
    location for two variables of interest.
    """
    def only_ktau(x, y):
        tau, _ = stats.kendalltau(x, y)
        return tau
    
    ktau = xr.apply_ufunc(
        only_ktau,
        ds[v1], ds[v2],
        input_core_dims=[['time'], ['time']],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True,
    ).rename('kendall_tau')
    

# %%