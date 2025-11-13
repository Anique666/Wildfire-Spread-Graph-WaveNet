#!/usr/bin/env python3
"""
check_and_viz.py

Checks the key aggregated outputs and visualizes grid coverage (which cells have panel observations).
Produces:
 - textual summaries
 - a plot showing grid cells colored by whether they have any panel rows and optionally observation counts.

Usage:
  python check_and_viz.py --agg_dir ./agg_output --preproc_dir ./preproc_output --plot

Requires: pandas, geopandas, matplotlib
"""
import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def summary_and_checks(agg_dir:Path, preproc_dir:Path):
    panel_p = agg_dir / "panel_cell_time.parquet"
    weather_p = agg_dir / "weather_cell_time.parquet"
    fire_p = agg_dir / "fire_cell_time.parquet"
    static_p = agg_dir / "cells_static.parquet"
    grid_p = preproc_dir / "grid.gpkg"

    # existence
    for p in (panel_p, weather_p, fire_p, static_p, grid_p):
        if not p.exists():
            print("MISSING:", p)
    panel = pd.read_parquet(panel_p)
    weather = pd.read_parquet(weather_p)
    fire = pd.read_parquet(fire_p)
    static = pd.read_parquet(static_p)
    grid = gpd.read_file(grid_p)

    print("\n=== Quick summaries ===")
    print("GRID: rows", len(grid), "CRS:", grid.crs)
    print("STATIC:", len(static), "unique cells:", static['cell_id'].nunique())
    print("WEATHER AGG:", len(weather), "unique cells:", weather['cell_id'].nunique(),
          "time range:", weather['datetime_bin'].min(), "->", weather['datetime_bin'].max())
    print("FIRE AGG:", len(fire), "unique cells:", fire['cell_id'].nunique(),
          "time range:", fire['datetime_bin'].min(), "->", fire['datetime_bin'].max())
    print("PANEL (sparse):", len(panel), "unique cells:", panel['cell_id'].nunique(),
          "time range:", panel['datetime_bin'].min(), "->", panel['datetime_bin'].max())

    # join grid with panel counts
    counts = pd.DataFrame(panel['cell_id'].value_counts()).reset_index()
    counts.columns = ["cell_id","obs_count"]
    grid_counts = grid.merge(counts, on="cell_id", how="left")
    grid_counts["obs_count"] = grid_counts["obs_count"].fillna(0).astype(int)
    # cells with any obs
    active_cells = grid_counts[grid_counts["obs_count"]>0].shape[0]
    print("Cells with any panel observations:", active_cells, "/", len(grid))

    return grid_counts, panel, weather, fire, static

def plot_grid_counts(grid_counts: gpd.GeoDataFrame, out_path:Path=None):
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    grid_counts.plot(column="obs_count", ax=ax, cmap="viridis", legend=True, linewidth=0.1)
    ax.set_title("Panel observation counts per grid cell (0 = no data)")
    ax.set_axis_off()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print("Saved plot to", out_path)
    plt.show()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agg_dir", required=True)
    p.add_argument("--preproc_dir", required=True)
    p.add_argument("--plot", action="store_true", help="Show & save plot")
    p.add_argument("--out_plot", default="./grid_panel_counts.png")
    args = p.parse_args()

    grid_counts, panel, weather, fire, static = summary_and_checks(Path(args.agg_dir), Path(args.preproc_dir))
    if args.plot:
        plot_grid_counts(grid_counts, Path(args.out_plot))

if __name__ == "__main__":
    main()
