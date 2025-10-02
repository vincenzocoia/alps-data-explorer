#!/usr/bin/env python3
"""
Example usage of the enhanced download function with aggregation
"""

from core import download_era5_data

# Example 1: Download data without aggregation
print("Example 1: Download without aggregation")
ds_raw = download_era5_data(
    polygon_coords=[[44.272, 8.274], [44.324, 7.494], [45.084, 7.305]],
    date_range=('2020-01-01', '2020-01-31'),
    agg_duration='none'
)
print(f"Raw data shape: {ds_raw.dims}")
print(f"Variables: {list(ds_raw.data_vars)}")

# Example 2: Download with daily aggregation
print("\nExample 2: Download with daily mean aggregation")
ds_daily = download_era5_data(
    polygon_coords=[[44.272, 8.274], [44.324, 7.494], [45.084, 7.305]],
    date_range=('2020-01-01', '2020-01-31'),
    agg_duration='daily',
    agg_func='mean'
)
print(f"Daily aggregated data shape: {ds_daily.dims}")

# Example 3: Download with monthly aggregation and specific variables
print("\nExample 3: Download with monthly max aggregation for specific variables")
ds_monthly = download_era5_data(
    polygon_coords=[[44.272, 8.274], [44.324, 7.494], [45.084, 7.305]],
    date_range=('2020-01-01', '2020-12-31'),
    agg_duration='monthly',
    agg_func='max',
    variables=['temperature_2m', 'total_precipitation']
)
print(f"Monthly aggregated data shape: {ds_monthly.dims}")
print(f"Variables: {list(ds_monthly.data_vars)}")

print("\nAll examples completed successfully!")
