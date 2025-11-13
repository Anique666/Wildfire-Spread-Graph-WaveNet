#!/usr/bin/env python3
"""
feature_agg.py (memory-footprint aware)

Purpose
-------
Memory-friendly feature aggregation for the wildfire MVP pipeline.
Reads the spatially-joined parquet files produced by data_pre.py:
  - topo_with_cell.parquet
  - weather_with_cell.parquet
  - fire_with_cell.parquet
  - grid.gpkg

Produces (saved to --out_dir):
  - cells_static.parquet         (static aggregated topo features per cell)
  - weather_cell_time.parquet   (weather aggregated per cell per time bin)
  - fire_cell_time.parquet      (fire aggregated per cell per time bin)
  - panel_cell_time.parquet     (SPARSE panel: only observed (cell, time) pairs merged)

Design considerations
---------------------
- **Sparse panel** only contains observed (cell_id, datetime_bin) pairs (from weather or fire),
  avoiding the full cartesian product (cells × times) which can explode memory.
- Aggregations are done with groupby and minimal intermediate copies.
- Forward/back-fill of weather values is done per-cell with groupby.apply to limit memory spikes.
- Optional flags allow writing intermediate chunks to disk if datasets are huge.

Usage example
-------------
python feature_agg.py \
  --preproc_dir ./preproc_output \
  --out_dir ./agg_output \
  --time_res_minutes 60 \
  --max_cells_in_memory 5000

Notes
-----
- Requires: pandas, numpy, geopandas
- For very large grids/time ranges consider increasing disk space and using the chunked options.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import warnings
warnings.simplefilter("ignore", FutureWarning)

# -------------------------
# Helper functions
# -------------------------
def safe_read_parquet(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)

def aggregate_static_topo(topo_df: pd.DataFrame, numeric_cols=None, cat_cols=None):
    """
    Aggregate static topo features by cell_id.
    Produces mean/std/min/max for numeric columns, and mode + proportions for categorical cols.
    """
    df = topo_df.copy()
    # If geometry column exists, drop for aggregation
    df = df.drop(columns=[c for c in df.columns if c == "geometry"], errors='ignore')

    # Infer numeric columns if not provided
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove lat/lon-like columns if present
        numeric_cols = [c for c in numeric_cols if c.lower() not in ("latitude","longitude","lon","lat","x","y")]

    agg_map = {}
    for c in numeric_cols:
        agg_map[c] = ["mean", "std", "min", "max"]
    grouped = df.groupby("cell_id").agg(agg_map)
    # flatten multiindex
    grouped.columns = ["_".join([col, func]) for col, func in grouped.columns]
    res = grouped.reset_index()

    # Categorical columns: mode and proportions
    if cat_cols:
        for c in cat_cols:
            # mode
            mode_series = df.groupby("cell_id")[c].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
            res = res.merge(mode_series.rename(f"{c}_mode").reset_index(), on="cell_id", how="left")
            # proportions per category value (sparse wide)
            counts = df.groupby(["cell_id", c]).size().unstack(fill_value=0)
            props = counts.div(counts.sum(axis=1), axis=0)
            props = props.add_prefix(f"{c}_prop_").reset_index()
            res = res.merge(props, on="cell_id", how="left")
    return res

def preprocess_weather_datetime(df: pd.DataFrame):
    """Ensure there's a datetime column called 'datetime' in the weather df."""
    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df
    # common patterns
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
        return df
    for c in ["acq_date","acq_time","timestamp","time"]:
        if c in df.columns:
            df["datetime"] = pd.to_datetime(df[c], errors="coerce")
            return df
    raise ValueError("Could not find/construct 'datetime' in weather dataframe. Please include date/time columns.")

def aggregate_weather_to_cells(weather_df: pd.DataFrame, time_res_minutes=60):
    """
    Aggregate weather by (cell_id, datetime_bin).
    Derives wind_speed and wind_dir if u10/v10 present.
    Returns a DataFrame with columns: cell_id, datetime_bin, <aggregates...>
    """
    df = weather_df.copy()
    df = preprocess_weather_datetime(df)
    df = df.drop(columns=[c for c in df.columns if c == "geometry"], errors='ignore')
    df = df.dropna(subset=["cell_id"])
    # floor to bin
    freq_str = f"{time_res_minutes}min"
    df["datetime_bin"] = df["datetime"].dt.floor(freq_str)

    # derive wind speed/dir
    if ("u10" in df.columns) and ("v10" in df.columns):
        df["u10"] = pd.to_numeric(df["u10"], errors="coerce")
        df["v10"] = pd.to_numeric(df["v10"], errors="coerce")
        df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
        df["wind_dir_rad"] = np.arctan2(df["v10"], df["u10"])
        df["wind_dir_deg"] = (np.degrees(df["wind_dir_rad"]) + 360) % 360

    # Choose aggregation map: adjust if your weather has different columns
    possible_cols = df.columns.tolist()
    agg_map = {}
    for c in ["t2m", "wind_speed", "u10", "v10", "wind_dir_deg"]:
        if c in possible_cols:
            # for temperature and wind speed we want mean/max/std
            if c in ("t2m","wind_speed"):
                agg_map[c] = ["mean", "max", "std"]
            else:
                agg_map[c] = ["mean"]
    # always include counts to know coverage
    df["_obs_count"] = 1
    agg_map["_obs_count"] = ["sum"]

    grouped = df.groupby(["cell_id", "datetime_bin"]).agg(agg_map)
    grouped.columns = ["_".join([col, func]) for col, func in grouped.columns]
    grouped = grouped.reset_index()
    return grouped

def aggregate_fire_to_cells(fire_df: pd.DataFrame, time_res_minutes=60):
    """
    Aggregate fire detections to (cell_id, datetime_bin):
    - frp_sum, frp_max, brightness_mean/max, fire_count, fire_any boolean
    """
    df = fire_df.copy()
    # try create datetime
    if "datetime" not in df.columns:
        if "acq_date" in df.columns and "acq_time" in df.columns:
            df["datetime"] = pd.to_datetime(df["acq_date"].astype(str) + " " + df["acq_time"].astype(str), errors="coerce")
        elif "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["datetime"] = pd.NaT

    df = df.drop(columns=[c for c in df.columns if c == "geometry"], errors='ignore')
    df = df.dropna(subset=["cell_id"])
    freq_str = f"{time_res_minutes}min"
    df["datetime_bin"] = df["datetime"].dt.floor(freq_str)

    agg_map = {}
    if "frp" in df.columns:
        agg_map["frp"] = ["sum", "max"]
    if "brightness" in df.columns:
        agg_map["brightness"] = ["mean", "max"]
    df["_dummy_count"] = 1
    agg_map["_dummy_count"] = ["sum"]

    grouped = df.groupby(["cell_id", "datetime_bin"]).agg(agg_map)
    grouped.columns = ["_".join([col, func]) for col, func in grouped.columns]
    grouped = grouped.reset_index().rename(columns={"_dummy_count_sum": "fire_count"})
    if "frp_sum" not in grouped.columns:
        grouped["frp_sum"] = grouped.get("frp_sum", 0)
    grouped["fire_any"] = grouped["fire_count"].fillna(0).astype(int) > 0
    # fill NaNs with zeros where appropriate
    grouped["fire_count"] = grouped["fire_count"].fillna(0).astype(int)
    for c in ["frp_sum", "frp_max", "brightness_mean", "brightness_max"]:
        if c in grouped.columns:
            grouped[c] = grouped[c].fillna(0)
    return grouped

def join_panel_sparse(weather_agg: pd.DataFrame, fire_agg: pd.DataFrame, grid_cells: gpd.GeoDataFrame,
                      time_res_minutes=60, build_dense=False, dense_chunk_size=1000, out_dir: Path=None):
    """
    Build a SPARSE panel: only (cell_id, datetime_bin) pairs that appear in either weather_agg or fire_agg.
    Returns merged panel with weather and fire columns. Forward/backfill weather per cell.

    If build_dense=True, creates full per-cell regular times but does it per-cell in chunks and writes
    chunked outputs to disk (controlled by dense_chunk_size and out_dir). Use only when necessary.
    """
    # ensure datetime types
    weather_agg = weather_agg.copy()
    fire_agg = fire_agg.copy()
    weather_agg["datetime_bin"] = pd.to_datetime(weather_agg["datetime_bin"])
    fire_agg["datetime_bin"] = pd.to_datetime(fire_agg["datetime_bin"])

    # 1) union of observed pairs (sparse)
    weather_pairs = weather_agg[["cell_id", "datetime_bin"]].drop_duplicates()
    fire_pairs = fire_agg[["cell_id", "datetime_bin"]].drop_duplicates()
    union = pd.concat([weather_pairs, fire_pairs], axis=0).drop_duplicates().reset_index(drop=True)

    # 2) left-merge aggregates
    panel = union.merge(weather_agg, on=["cell_id", "datetime_bin"], how="left", suffixes=("", "_w"))
    panel = panel.merge(fire_agg, on=["cell_id", "datetime_bin"], how="left", suffixes=("", "_f"))

    # 3) sanitize fire columns
    if "fire_count" in panel.columns:
        panel["fire_count"] = panel["fire_count"].fillna(0).astype(int)
    else:
        panel["fire_count"] = 0
    if "fire_any" in panel.columns:
        panel["fire_any"] = panel["fire_any"].fillna(False).astype(bool)
    else:
        panel["fire_any"] = panel["fire_count"] > 0
    # fill numeric columns that are expected
    for c in panel.columns:
        if c.endswith("_sum") or c.endswith("_max") or c.endswith("_mean"):
            panel[c] = panel[c].fillna(0)

    # 4) forward/backfill weather per cell for continuity (use groupby apply to limit memory)
    weather_cols = [c for c in panel.columns if any(s in c for s in ["t2m", "wind_speed", "u10", "v10", "wind_dir_deg"])]
    if weather_cols:
        panel = panel.sort_values(["cell_id", "datetime_bin"])
        # groupby.apply can be heavyweight if many groups; we do it, but it's streaming-friendly
        def _ffill_bfill(g):
            g[weather_cols] = g[weather_cols].ffill().bfill()
            return g
        panel = panel.groupby("cell_id", group_keys=False).apply(_ffill_bfill).reset_index(drop=True)

    # 5) Optionally build dense panel per cell (memory-hungry) in controlled chunks
    if build_dense:
        if out_dir is None:
            raise ValueError("out_dir must be provided when build_dense=True to write chunked outputs.")
        print("Building dense panel per cell (chunked). This may take time and disk space.")
        cells = grid_cells["cell_id"].unique()
        freq_str = f"{time_res_minutes}min"
        out_parts = []
        for i, cid in enumerate(cells):
            subset = panel[panel["cell_id"] == cid]
            if subset.empty:
                continue
            min_t = subset["datetime_bin"].min()
            max_t = subset["datetime_bin"].max()
            times_reg = pd.date_range(min_t, max_t, freq=freq_str)
            df_times = pd.DataFrame({"cell_id": cid, "datetime_bin": times_reg})
            merged = df_times.merge(subset, on=["cell_id", "datetime_bin"], how="left")
            # fill weather via forward/backfill
            if weather_cols:
                merged[weather_cols] = merged[weather_cols].ffill().bfill()
            # write chunk to disk periodically
            out_parts.append(merged)
            if len(out_parts) >= dense_chunk_size:
                part_df = pd.concat(out_parts, axis=0, ignore_index=True)
                part_path = out_dir / f"panel_dense_part_{i}.parquet"
                part_df.to_parquet(part_path)
                print("Wrote chunk:", part_path)
                out_parts = []
        # final flush
        if out_parts:
            part_df = pd.concat(out_parts, axis=0, ignore_index=True)
            part_path = out_dir / f"panel_dense_part_final.parquet"
            part_df.to_parquet(part_path)
            print("Wrote final dense chunk:", part_path)
        # if build_dense True we don't return a huge DataFrame
        return None

    return panel

# -------------------------
# Main script
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", required=True, help="Directory produced by data_pre.py")
    p.add_argument("--out_dir", required=True, help="Directory to write aggregated outputs")
    p.add_argument("--time_res_minutes", type=int, default=60, help="Time bin size in minutes (default=60)")
    p.add_argument("--build_dense", action="store_true", help="(Dangerous) build dense panel for every cell x timestep (writes chunked files)")
    p.add_argument("--dense_chunk_size", type=int, default=1000, help="Chunk size for dense panel writes (only with --build_dense)")
    p.add_argument("--max_cells_in_memory", type=int, default=5000, help="If >0, will warn when grid has > this many cells")
    args = p.parse_args()

    preproc = Path(args.preproc_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # input paths
    topo_path = preproc / "topo_with_cell.parquet"
    weather_path = preproc / "weather_with_cell.parquet"
    fire_path = preproc / "fire_with_cell.parquet"
    grid_path = preproc / "grid.gpkg"

    print("Loading preprocessed files...")
    topo = safe_read_parquet(topo_path)
    weather = safe_read_parquet(weather_path)
    fire = safe_read_parquet(fire_path)
    grid = gpd.read_file(grid_path)

    print("Aggregating static topo features per cell...")
    # infer numeric and categorical columns
    topo_cols = topo.drop(columns=[c for c in topo.columns if c == "geometry"], errors='ignore')
    numeric_cols = topo_cols.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() not in ("latitude","longitude","lon","lat","x","y")]
    cat_cols = topo_cols.select_dtypes(exclude=[np.number]).columns.tolist()
    # remove cell_id and obvious non-feature columns
    cat_cols = [c for c in cat_cols if c not in ("cell_id", "geometry", "centroid")] 
    static_agg = aggregate_static_topo(topo, numeric_cols=numeric_cols, cat_cols=cat_cols)
    static_out = out_dir / "cells_static.parquet"
    static_agg.to_parquet(static_out)
    print("Saved static:", static_out)

    print("Aggregating weather to cells x time bins...")
    weather = weather.drop(columns=[c for c in weather.columns if c == "geometry"], errors='ignore')
    weather_agg = aggregate_weather_to_cells(weather, time_res_minutes=args.time_res_minutes)
    weather_out = out_dir / "weather_cell_time.parquet"
    weather_agg.to_parquet(weather_out)
    print("Saved weather agg:", weather_out)

    print("Aggregating fire detections to cells x time bins...")
    fire = fire.drop(columns=[c for c in fire.columns if c == "geometry"], errors='ignore')
    fire_agg = aggregate_fire_to_cells(fire, time_res_minutes=args.time_res_minutes)
    fire_out = out_dir / "fire_cell_time.parquet"
    fire_agg.to_parquet(fire_out)
    print("Saved fire agg:", fire_out)

    # advisory about grid size
    n_cells = grid["cell_id"].nunique()
    if args.max_cells_in_memory and n_cells > args.max_cells_in_memory:
        print(f"WARNING: grid has {n_cells} cells which is > max_cells_in_memory ({args.max_cells_in_memory}).")
        print("Using sparse panel mode (default). If you need dense panel consider clustering or using --build_dense with chunking.")

    print("Joining into SPARSE panel (only observed cell x time pairs)...")
    panel = join_panel_sparse(weather_agg, fire_agg, grid,
                              time_res_minutes=args.time_res_minutes,
                              build_dense=args.build_dense,
                              dense_chunk_size=args.dense_chunk_size,
                              out_dir=out_dir if args.build_dense else None)
    if args.build_dense:
        print("Dense panel build requested: parts written to out_dir (see messages). Exiting.")
        return

    panel_out = out_dir / "panel_cell_time.parquet"
    panel.to_parquet(panel_out)
    print("Saved panel:", panel_out)
    print("Panel size (rows):", len(panel))
    print("Unique cells in panel:", panel['cell_id'].nunique())
    print("Time span:", panel['datetime_bin'].min(), "->", panel['datetime_bin'].max())

    print("All done. Next: feature engineering (lags, neighbor aggregates, labels).")

if __name__ == "__main__":
    main()
