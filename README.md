# Alps Data Explorer

The purpose of this repository is to be a data exploration playground for data related to the project "Decoding Rain-on-Snow Flooding using Statistics and Satellite Data". See more about this project at the project website, <https://rainonsnow.vincenzocoia.com/>.

## Dependencies

I'm using python for obtaining the data and visualizing it. I've created a conda environment that you can reproduce from the `environment.yml` file by executing the command:

```
conda env create -f environment.yml
```

The environment is named `alps-data-explorer`, like this repository.

## Workflow

The main idea behind the codebase:

- Access satellite data programmatically using Google Earth Engine and XEE.
- Wrangle the data using xarray.
- Display slices of data by deploying an interactive streamlit app, where users can specify what "slice" of the data they want (what pixel? what data aggregation -- max? What time range?)

## Project Structure

The codebase follows a modular architecture for both interactive exploration and web deployment:

```
├── config.py          # Centralized configuration and defaults
├── core.py            # Core analysis functions (UI-agnostic)
├── app.py             # Clean Streamlit web application
├── explore.ipynb      # Jupyter notebook for interactive exploration
├── utils/             # Statistical utility functions
│   ├── __init__.py
│   └── stats.py       # Custom statistical functions (nscore, etc.)
└── environment.yml    # Conda environment specification
```

## Usage

### Interactive Exploration (Recommended)
```bash
# Launch Jupyter for data exploration
jupyter lab explore.ipynb
```

### Web Application
```bash
# Launch Streamlit web app
streamlit run app.py
```

## Contributing

This project is open source and welcomes contributions. Please see the [contributing guide](CONTRIBUTING) and the [code of conduct](CODE_OF_CONDUCT).