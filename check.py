import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

grid = gpd.read_file("./preproc_output/grid.gpkg")
panel = pd.read_parquet("./agg_output/panel_cell_time.parquet")
panel_cells = panel["cell_id"].unique()

grid["in_panel"] = grid["cell_id"].isin(panel_cells)
ax = grid.plot(column="in_panel", figsize=(8,8), legend=True, categorical=True)
plt.title("Grid cells with any data in PANEL (colored)")
plt.show()
