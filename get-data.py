# %%
# Import necessary libraries
import ee
import xarray as xr
import xrspatial as xs
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium

st.title('Alps Data Explorer')


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
folium.Polygon(alps_polygon_coords, color='blue', fill=True, fill_opacity=0.2).add_to(m)

# Display the map
st_folium(m, width=700, height=500)

# %%
# Download some ERA5 data
ic = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate(
    '1992-10-05', '1993-03-31')
leg1 = ee.Geometry.Polygon(alps_polygon_coords)
ds = xr.open_dataset(
    ic,
    #engine='netcdf4',
    projection=ic.first().select(0).projection(),
    geometry=leg1
)

# %%
import cartopy.crs as ccrs

ro = ds.runoff_hourly.isel(time=0)

p = ro.plot(
    subplot_kws=dict(
        projection=ccrs.epsg(3857),  # Web Mercator projection
        facecolor="gray"
    ),
    transform=ccrs.PlateCarree(),  # Use PlateCarree for geographic data
)

# Set the extent to focus on the Alps
# Example extent: [min_lon, max_lon, min_lat, max_lat]
p.axes.set_extent([0, 20, 40, 50], crs=ccrs.PlateCarree())

# Add coastlines and other features
p.axes.coastlines()

st.pyplot(p.figure)

# %%

