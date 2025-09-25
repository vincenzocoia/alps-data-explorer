# Alps Data Explorer

The purpose of this repository is to be a data exploration playground for data related to the project "Decoding Rain-on-Snow Flooding using Statistics and Satellite Data".

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