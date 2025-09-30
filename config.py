"""
Configuration for Alps Data Explorer
Centralized settings for both interactive and web app modes
"""

# Default values for interactive mode
DEFAULT_CONFIG = {
    'v1': 'snowmelt',
    'v2': 'surface_runoff_hourly', 
    'agg_duration': 'monthly',
    'agg_func': 'mean',
    'scale': 'Linear',
    'polygon_coords': [
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
    ],
    'date_range': ('1992-10-05', '1993-03-31'),
    'ee_project': 'alps-data-explorer'
}

# Widget options for Streamlit
WIDGET_OPTIONS = {
    'agg_duration_options': ['monthly', 'yearly', 'daily'],
    'agg_func_options': ['max', 'min', 'mean', 'sum'],
    'scale_options': ['Linear', 'Normal Scores']
}
