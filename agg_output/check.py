#!/usr/bin/env python3
"""
check_feature_agg.py

Quick diagnostic script to verify outputs of feature_agg.py
Checks:
 - File presence
 - Basic shapes, columns, time coverage
 - NaN ratios
 - Grid sanity
Optionally plots time distribution and grid coverage.

Usage:
    python check_feature_agg.py --agg_dir ./agg_output --preproc_dir ./preproc_output
"""
import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def print_header(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def describe_df(name, df, time_col=None, ncols=10):
    print_header(f"{name}")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Sample columns:", df.columns[:ncols].tolist())
    if "cell_id" in df.columns:
        print("Unique cells:", df["cell_id"].nunique())
    if time_col and time_col in df.columns:
        print("Time span:", df[time_col].min(), "->", df[time_col].max())
        print("Unique time bins:", df[time_col].nunique())
    print("NaN counts (top 10 cols):")
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print("-"*60)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agg_dir", required=True, help="Directory containing feature_agg.py outputs")
    p.add_argument("--preproc_dir", required=True, help="Directory containing grid.gpkg")
    p.add_argument("--plot", action="store_true", help="Show optional quick visual plots")
    args = p.parse_args()

    agg_dir = Path(args.agg_dir)
    preproc_dir = Path(args.preproc_dir)

    # paths
    panel_path = agg_dir / "panel_cell_time.parquet"
    weather_path = agg_dir / "weather_cell_time.parquet"
    fire_path = agg_dir / "fire_cell_time.parquet"
    static_path = agg_dir / "cells_static.parquet"
    grid_path = preproc_dir / "grid.gpkg"

    # load and describe each
    print_header("Checking aggregated outputs")

    if not panel_path.exists():
        print("❌ Missing:", panel_path); return
    if not weather_path.exists():
        print("❌ Missing:", weather_path); return
    if not fire_path.exists():
        print("❌ Missing:", fire_path); return
    if not static_path.exists():
        print("❌ Missing:", static_path); return
    if not grid_path.exists():
        print("❌ Missing:", grid_path); return

    panel = pd.read_parquet(panel_path)
    weather = pd.read_parquet(weather_path)
    fire = pd.read_parquet(fire_path)
    static = pd.read_parquet(static_path)
    grid = gpd.read_file(grid_path)

    describe_df("GRID", grid)
    print("CRS:", grid.crs)
    print("Total cells:", len(grid))

    describe_df("STATIC FEATURES", static)
    describe_df("WEATHER AGGREGATES", weather, time_col="datetime_bin")
    describe_df("FIRE AGGREGATES", fire, time_col="datetime_bin")
    describe_df("PANEL (sparse merged)", panel, time_col="datetime_bin")

    # sanity checks
    print_header("Quick sanity checks")
    cells_panel = set(panel["cell_id"].unique())
    cells_static = set(static["cell_id"].unique())
    overlap = len(cells_panel.intersection(cells_static))
    print(f"Cells overlap (panel ∩ static): {overlap} / {len(cells_panel)}")
    print("Panel covers {:.2f}% of static cells".format(100 * overlap / max(1, len(cells_static))))

    # time resolution
    if "datetime_bin" in panel.columns:
        tdiff = panel.sort_values("datetime_bin")["datetime_bin"].diff().dropna().dt.total_seconds()
        avg_diff = tdiff.mean() / 3600 if len(tdiff) else None
        print("Average temporal spacing (hours):", round(avg_diff, 2) if avg_diff else "N/A")

    # missing values overview
    na_summary = panel.isna().mean().sort_values(ascending=False)
    print("\nTop 15 columns with most missing values in PANEL:")
    print(na_summary.head(15))

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            print_header("Plotting quick diagnostics")
            # 1. time distribution
            panel["datetime_bin"].hist(bins=50)
            plt.title("Distribution of datetime_bin in panel")
            plt.xlabel("Datetime bin")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()

            # 2. spatial coverage
            if "cell_id" in panel.columns:
                cells = pd.DataFrame(panel["cell_id"].value_counts()).reset_index()
                cells.columns = ["cell_id", "obs_count"]
                gjoin = grid.merge(cells, on="cell_id", how="left")
                gjoin["obs_count"] = gjoin["obs_count"].fillna(0)
                gjoin.plot(column="obs_count", cmap="viridis", legend=True)
                plt.title("Observation counts per cell in panel")
                plt.axis("off")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print("Plotting failed:", e)

    print("\n✅ All checks completed. If no ❌ appeared, outputs look good.")

if __name__ == "__main__":
    main()
