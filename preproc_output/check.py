import geopandas as gpd
grid = gpd.read_file("./preproc_output/grid.gpkg")
print(grid.head())
print(grid.crs)          # Should show a projected CRS, e.g. EPSG:32643
print(len(grid), "cells")

# Optional: quick visualization
grid.plot(edgecolor='grey', facecolor='none')
import pandas as pd
topo = pd.read_parquet("./preproc_output/topo_with_cell.parquet")
print(topo.shape)
print(topo[['longitude', 'latitude', 'cell_id']].head())
print("Unique grid cells covered:", topo['cell_id'].nunique())
weather = pd.read_parquet("./preproc_output/weather_with_cell.parquet")
print(weather.shape)
print(weather[['latitude','longitude','cell_id']].head())
print(weather.columns)
fire = pd.read_parquet("./preproc_output/fire_with_cell.parquet")
print(fire.shape)
print(fire[['latitude','longitude','cell_id']].head())
print(fire.columns)
centroids = pd.read_csv("./preproc_output/grid_centroids.csv")
print(centroids.head())
print(len(centroids), "centroids in total")
