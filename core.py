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
import contextily as ctx
import utils
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
        The Earth Engine project to use.
        
    Returns
    -------
    None
    """
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
    # Indicate that no aggregation has been applied (as metadata)
    ds.attrs['transformation'] = None
    return ds

# %%
def xr_transform_nscore(ds: xr.Dataset) -> xr.Dataset:
    """
    Transform the dataset to normal scores.
    
    Inputs:
    - ds: xarray dataset with two variables of interest.
    
    Returns:
    - xarray dataset with the two variables transformed to normal scores.
    """
    ds = ds.apply(utils.nscore)
    
    # Update metadata
    if ds.attrs['transformation']:
        ds.attrs['transformation'] = 'nscore ∘ ' + ds.attrs['transformation']
    else:
        ds.attrs['transformation'] = 'nscore'
    
    return ds

# %%
def xr_transform_aggregate(ds: xr.Dataset,
                           agg_duration: str,
                           agg_func: str) -> xr.Dataset:
    """
    Apply an aggregating function to the dataset for a specified duration.
    
    Inputs:
    - ds: xarray dataset with two variables of interest.
    - agg_duration: duration of the aggregation. One of 'daily', 'monthly', 
      'yearly', or 'none'. If 'none', no aggregation is applied, so that
      `agg_func` is ignored.
    - agg_func: function to apply to the aggregation.
    
    Returns:
    - xarray dataset with the two variables aggregated.
    
    Raises:
    - ValueError: if agg_duration is not one of the supported values.
    """
    # Mapping from user-friendly names to pandas frequency strings
    duration_mapping = {
        'daily': 'D',
        'monthly': 'ME', 
        'yearly': 'Y',
        'none': None
    }
    
    # Validate input
    if agg_duration not in duration_mapping:
        valid_options = list(duration_mapping.keys())
        raise ValueError(
            f"agg_duration must be one of {valid_options}, got '{agg_duration}'"
        )
    
    # Validate aggregation function exists
    if agg_duration != 'none' and not hasattr(np, agg_func):
        raise ValueError(f"agg_func '{agg_func}' is not a valid numpy function")
    
    # Apply aggregation based on duration
    freq_code = duration_mapping[agg_duration]
    if agg_duration != 'none':
        ds = ds.resample(time=freq_code).reduce(getattr(np, agg_func))
        if ds.attrs['transformation']:
            ds.attrs['transformation'] = f'{agg_duration}_{agg_func} ∘ ' + \
                ds.attrs['transformation']
        else:
            ds.attrs['transformation'] = f'{agg_duration}_{agg_func}'
        
    return ds

# %%
def pluck_v1v2(ds: xr.Dataset, 
               v1: str, 
               v2: str) -> xr.Dataset:
    """
    Zero-in on two variables of interest in the xarray dataset, with an 
    aggregation function applied for a specified duration.
    
    Inputs:
    - ds: xarray dataset with two variables of interest.
    - v1: name of the first variable of interest.
    - v2: name of the second variable of interest.
    - agg_duration: duration of the aggregation. One of 'daily', 'monthly', 
      'yearly', or 'none'. If 'none', no aggregation is applied, so that
      `agg_func` is ignored.
    - agg_func: function to apply to the aggregation.
    
    """
    ds = ds[[v1, v2]]
    return ds

# %%
def pluck_xy(ds: xr.Dataset,
             lon: float,
             lat: float) -> Dict[str, Any]:
    """
    Parse out the numpy arrays of the xarray variables over time, at a specific
    point. Also return normal scores of the variables.
    
    Inputs:
    - ds: xarray dataset with two variables of interest.
    - lon: longitude coordinate.
    - lat: latitude coordinate.

    Returns:
    - xarray dataset with only the lat and lon coordinates nearest to the
      specified lat and lon coordinates.
    """
    ds_xy = ds.sel(lon=lon, lat=lat, method='nearest')
    return ds_xy

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
    - biv_dependence_data: xarray dataset with two variables of interest
      and scalar (single) lat-lon coordinates.
    
    Returns:
    - dictionary of two time series plots, one for each variable of interest.
    """
    assert len(biv_xarray.data_vars) == 2, "Bivariate xarray must have two data variables."
    assert biv_xarray.lat.ndim == 0 and biv_xarray.lon.ndim == 0, "Bivariate xarray must have scalar lat-lon coordinates."
    v1, v2 = list(biv_xarray.data_vars)
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
                               use_nscore: bool = False) -> plt.Figure:
    """
    Create matplotlib scatter plot of two variables of interest from
    the bivariate dependence results, `biv_dependence_data()`.
    
    Inputs:
    - biv_dependence_data: xarray dataset with two variables of interest
      and scalar (single) lat-lon coordinates.
    - use_nscore: boolean indicating whether to use normal scores for the 
      scatter plot.
    
    Returns:
    - scatter plot of two variables of interest.
    """
    assert len(biv_xarray.data_vars) == 2, "Bivariate xarray must have two data variables."
    assert biv_xarray.lat.ndim == 0 and biv_xarray.lon.ndim == 0, "Bivariate xarray must have scalar lat-lon coordinates."
    v1, v2 = list(biv_xarray.data_vars)
    v1_data = biv_xarray[v1].values
    v2_data = biv_xarray[v2].values
    rho = stats.kendalltau(v1_data, v2_data).correlation
    scale = 'Linear'
    if use_nscore:
        v1_data = utils.nscore(v1_data)
        v2_data = utils.nscore(v2_data)
        scale = 'Normal Scores'

    # Create the scatter plot.
    fig_scatter = plt.figure(figsize=(8, 6))
    plt.scatter(v1_data, v2_data)
    plt.xlabel(v1)
    plt.ylabel(v2)
    plt.title(f'Scatter plot. Scale: {scale}. Kendall\'s tau: {rho:.2f}')
    
    return fig_scatter

# %%
def xr_transform_ktau(biv_xarray: Dict[str, Any]) -> xr.Dataset:
    """
    Create xarray dataset of Kendall's tau correlations for each (x, y)
    location for two variables of interest.
    
    Inputs:
    - biv_xarray: xarray dataset of two variables of interest.
    
    Returns:
    - xarray data array of Kendall's tau correlations for each (x, y)
    location for two variables of interest.
    """
    assert len(biv_xarray.data_vars) == 2, "Bivariate xarray must have two data variables."
    v1, v2 = list(biv_xarray.data_vars)
    def only_ktau(x, y):
        tau, _ = stats.kendalltau(x, y)
        return tau
    
    ktau = xr.apply_ufunc(
        only_ktau,
        biv_xarray[v1], biv_xarray[v2],
        input_core_dims=[['time'], ['time']],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True,
    ).rename('kendall_tau')
    
    # Update metadata
    if biv_xarray.attrs.get('transformation'):
        ktau.attrs['transformation'] = 'kendall_tau ∘ ' + \
            biv_xarray.attrs['transformation']
    else:
        ktau.attrs['transformation'] = 'kendall_tau'
    
    return ktau

# %%
def xr_heatmap(dataarray: xr.DataArray, 
               basemap=ctx.providers.Esri.WorldTerrain):
    """
    Create heatmap with basemap behind the data.
    
    Inputs:
    - dataarray: xarray dataarray with only lat and lon dimensions and no time.
    - basemap: contextily basemap provider (default: Esri.WorldTerrain)
    
    Returns:
    - (fig, ax): matplotlib Figure and Axes objects for further customization
    """
    # Assert that it's a dataarray with only lat and lon dimensions (no time)
    expected_dims = {'lat', 'lon'}
    actual_dims = set(dataarray.dims)
    assert actual_dims == expected_dims, f"Dataarray must have only lat and lon dimensions. Got: {dataarray.dims}"

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the heatmap as pixels (not contours) FIRST
    dataarray.plot.pcolormesh(ax=ax, cmap='viridis', alpha=0.7, add_colorbar=True)
    
    # Get current axis limits after plotting data
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Add a 20% buffer
    xlim = (xlim[0] - 0.2 * (xlim[1] - xlim[0]), xlim[1] + 0.2 * (xlim[1] - xlim[0]))
    ylim = (ylim[0] - 0.2 * (ylim[1] - ylim[0]), ylim[1] + 0.2 * (ylim[1] - ylim[0]))
    
    # Add basemap behind the data
    try:
        ctx.add_basemap(ax, crs='EPSG:4326', source=basemap)
        
        # Restore limits (basemap might change them)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
    except Exception as e:
        ax.set_facecolor('lightgray')
    
    return fig, ax


# %%
