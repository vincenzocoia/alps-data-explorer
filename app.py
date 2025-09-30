"""
Streamlit web app for Alps Data Explorer
Clean separation between UI and business logic
"""

import streamlit as st
from streamlit_folium import st_folium
from core import initialize_earth_engine, download_era5_data, run_analysis, create_alps_map
from config import DEFAULT_CONFIG, WIDGET_OPTIONS

# Page config
st.set_page_config(page_title="Alps Data Explorer", layout="wide")
st.title('Alps Data Explorer')

# Initialize Earth Engine (cached)
@st.cache_resource
def init_ee():
    initialize_earth_engine()
    return True

@st.cache_data
def load_data():
    return download_era5_data()

# Initialize
init_ee()
ds = load_data()

# Layout
col1, col2 = st.columns(2)

# Display map
with col1:
    st.subheader("Study Area")
    m = create_alps_map()
    st_folium(m, width=700, height=500)

# Sidebar controls
st.sidebar.header("Analysis Parameters")

v = st.sidebar.selectbox("Select first variable", ds.data_vars, 
                        index=list(ds.data_vars).index(DEFAULT_CONFIG['v']) 
                        if DEFAULT_CONFIG['v'] in ds.data_vars else 0)

v2 = st.sidebar.selectbox("Select second variable", ds.data_vars,
                         index=list(ds.data_vars).index(DEFAULT_CONFIG['v2']) 
                         if DEFAULT_CONFIG['v2'] in ds.data_vars else 1)

choose_x = st.sidebar.select_slider("Longitude", ds.lon.values, 
                                   value=ds.lon.values[len(ds.lon.values)//2])

choose_y = st.sidebar.select_slider("Latitude", ds.lat.values,
                                   value=ds.lat.values[len(ds.lat.values)//2])

agg_duration = st.sidebar.selectbox("Aggregation duration", 
                                   WIDGET_OPTIONS['agg_duration_options'],
                                   index=WIDGET_OPTIONS['agg_duration_options'].index(DEFAULT_CONFIG['agg_duration']))

agg_func = st.sidebar.selectbox("Aggregation function",
                               WIDGET_OPTIONS['agg_func_options'],
                               index=WIDGET_OPTIONS['agg_func_options'].index(DEFAULT_CONFIG['agg_func']))

scale = st.sidebar.radio("Scale", WIDGET_OPTIONS['scale_options'],
                        index=WIDGET_OPTIONS['scale_options'].index(DEFAULT_CONFIG['scale']))

# Analysis button
if st.sidebar.button("Run Analysis", type="primary"):
    with st.spinner("Running analysis..."):
        # Run analysis
        results, plots = run_analysis(
            v=v, v2=v2, choose_x=choose_x, choose_y=choose_y,
            agg_duration=agg_duration, agg_func=agg_func, scale=scale, ds=ds
        )
        
        # Display results
        st.success(f"Analysis complete! Location: ({choose_x:.2f}, {choose_y:.2f})")
        
        # Show plots
        with col2:
            st.subheader("Scatter Plot")
            st.pyplot(plots['scatter'])
        
        with col1:
            st.subheader(f"Time Series: {v}")
            st.pyplot(plots['timeseries1'])
        
        with col2:
            st.subheader(f"Time Series: {v2}")
            st.pyplot(plots['timeseries2'])
        
        # Show correlation map
        st.subheader("Spatial Correlation Map")
        tau_data = results['tau'].values
        if not all(np.isnan(tau_data.flatten())):
            fig_map = results['tau'].plot(figsize=(12, 8), cmap='RdBu_r', center=0)
            st.pyplot(fig_map.figure)
        else:
            st.warning("No valid correlation data to display")

# Info sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app explores relationships between climate variables in the Alps region.
    
    **Data**: ERA5-Land hourly reanalysis  
    **Period**: Oct 1992 - Mar 1993  
    **Method**: Kendall's tau correlation
    """)
