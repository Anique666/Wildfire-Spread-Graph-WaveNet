#!/usr/bin/env python3
"""
postproc_agg_fixed.py

Post-process outputs of feature_agg.py with three steps:
  A) Re-assign weather to grid cells by nearest ORIGINAL weather station (from CSV).
  B) Ensure only panel-active cells can be marked 'fire_any' = True (filter other fire rows).
  C) Impute missing static/topo features using k-NN (k=8 by default).

Inputs (expected):
  - <agg_dir>/panel_cell_time.parquet
  - <agg_dir>/fire_cell_time.parquet
  - <agg_dir>/cells_static.parquet
  - <preproc_dir>/grid_centroids.csv
  - <preproc_dir>/grid.gpkg
  - <weather_csv> (raw CSV: output_final_temp_celsius_fixed.csv)

Outputs (overwrites/creates):
  - <agg_dir>/weather_cell_time.parquet (new)
  - <agg_dir>/fire_cell_time.parquet (filtered to panel cells)
  - <agg_dir>/cells_static.parquet (imputed)
  - <agg_dir>/panel_cell_time.parquet (panel fire columns replaced with cleaned fire)

Usage example:
  python postproc_agg_fixed.py \
    --agg_dir ./agg_output \
    --preproc_dir ./preproc_output \
    --weather_csv ./output_final_temp_celsius_fixed.csv \
    --time_res_minutes 60 \
    --k 8

Requires: pandas, geopandas, numpy, scikit-learn
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.neighbors import BallTree, NearestNeighbors
import warnings
warnings.simplefilter("ignore", FutureWarning)

# -------------------
# Helpers
# -------------------
def guess_latlon_cols(df):
    lat = next((c for c in df.columns if c.lower() in ("lat","latitude","y","centroid_y")), None)
    lon = next((c for c in df.columns if c.lower() in ("lon","longitude","x","centroid_x")), None)
    # fallback to substring match
    if lat is None:
        lat = next((c for c in df.columns if "lat" in c.lower()), None)
    if lon is None:
        lon = next((c for c in df.columns if "lon" in c.lower() or "long" in c.lower()), None)
    if lat is None or lon is None:
        raise ValueError(f"Could not find lat/lon columns in DataFrame. Columns: {df.columns.tolist()}")
    return lat, lon

def ensure_datetime(df, possible_cols=None):
    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df
    # try common pairs
    if possible_cols:
        if isinstance(possible_cols, (list,tuple)) and len(possible_cols) == 2:
            dcol, tcol = possible_cols
            if dcol in df.columns and tcol in df.columns:
                df["datetime"] = pd.to_datetime(df[dcol].astype(str) + " " + df[tcol].astype(str), errors="coerce")
                return df
    for c in ("datetime","timestamp","time","date","acq_date"):
        if c in df.columns:
            df["datetime"] = pd.to_datetime(df[c], errors="coerce")
            return df
    # try date+time names
    if "acq_date" in df.columns and "acq_time" in df.columns:
        df["datetime"] = pd.to_datetime(df["acq_date"].astype(str) + " " + df["acq_time"].astype(str), errors="coerce")
        return df
    # last resort: try parsing index
    try:
        df["datetime"] = pd.to_datetime(df.index, errors="coerce")
        return df
    except Exception:
        raise ValueError("Could not infer datetime column; please provide parseable date/time columns.")

# -------------------
# Step A: remap weather using nearest station
# -------------------
def remap_weather(grid_centroids_csv:Path, weather_csv:Path, out_weather_parquet:Path, time_res_minutes:int=60):
    print("STEP A: Remapping weather to nearest raw station per grid cell.")
    grid_cent = pd.read_csv(grid_centroids_csv)
    lat_col_grid, lon_col_grid = guess_latlon_cols(grid_cent)
    # ensure numeric
    grid_cent[[lat_col_grid, lon_col_grid]] = grid_cent[[lat_col_grid, lon_col_grid]].astype(float)

    print("Loading raw weather CSV (may take a moment)...")
    raw_w = pd.read_csv(weather_csv)
    lat_w, lon_w = guess_latlon_cols(raw_w)
    print(f"Detected weather lat/lon columns: {lat_w}, {lon_w}")
    raw_w = ensure_datetime(raw_w)
    raw_w = raw_w.dropna(subset=["datetime"])
    # unique stations by lat/lon
    station_keys = raw_w[[lat_w, lon_w]].drop_duplicates().reset_index(drop=True).rename(columns={lat_w:"station_lat", lon_w:"station_lon"})
    station_keys = station_keys.reset_index().rename(columns={"index":"station_idx"})
    print("Unique stations:", len(station_keys))

    # Build BallTree on station coords (lat, lon) in radians (haversine)
    def to_rad(arr):
        return np.deg2rad(arr.astype(float))
    station_coords = station_keys[["station_lat","station_lon"]].values.astype(float)
    station_rad = to_rad(station_coords)
    tree = BallTree(station_rad, metric="haversine")

    # prepare grid coords (lat, lon) for query
    grid_coords = grid_cent[[lat_col_grid, lon_col_grid]].values.astype(float)
    grid_rad = to_rad(grid_coords)
    dist_rad, idx = tree.query(grid_rad, k=1)
    earth_r = 6371000.0
    dist_m = (dist_rad.flatten() * earth_r)
    grid_cent["nearest_station_idx"] = idx.flatten()
    grid_cent["nearest_station_dist_m"] = dist_m

    # attach station info to grid rows
    station_map = station_keys.set_index("station_idx")[["station_lat","station_lon"]]
    # merge raw_w with station_keys to tag each row with station_idx
    raw_w = raw_w.merge(station_keys, left_on=[lat_w, lon_w], right_on=["station_lat","station_lon"], how="left")
    if raw_w["station_idx"].isna().any():
        print("Warning: some weather rows could not be linked to station_keys by exact lat/lon match. Those rows are dropped.")
    raw_w = raw_w.dropna(subset=["station_idx"])
    raw_w["station_idx"] = raw_w["station_idx"].astype(int)

    freq_str = f"{time_res_minutes}min"
    raw_w["datetime_bin"] = pd.to_datetime(raw_w["datetime"]).dt.floor(freq_str)

    # detect weather numeric columns to aggregate
    possible_cols = [c for c in raw_w.columns if c not in (lat_w, lon_w, "datetime", "datetime_bin", "station_idx","station_lat","station_lon")]
    # pick common columns if present
    cand = [c for c in ("t2m","u10","v10","air_temperature","temperature","t2m_c") if c in raw_w.columns]
    if cand:
        agg_cols = cand
    else:
        # fallback: numeric columns excluding station/coords
        agg_cols = [c for c in raw_w.select_dtypes(include=np.number).columns if c not in ("station_idx",)]
    print("Aggregating these weather columns:", agg_cols)

    # aggregate per station per time bin
    agg_map = {c:["mean","max"] for c in agg_cols}
    raw_w["_cnt"] = 1
    agg_map["_cnt"] = ["sum"]
    station_time = raw_w.groupby(["station_idx","datetime_bin"]).agg(agg_map)
    station_time.columns = ["_".join([col, func]) for col, func in station_time.columns]
    station_time = station_time.reset_index()

    # map station_time to cells by nearest_station_idx -> station_idx
    # build mapping of station_idx -> list(cell_id)
    mapping = grid_cent.groupby("nearest_station_idx")["cell_id"].agg(list).reset_index().rename(columns={"nearest_station_idx":"station_idx"})
    # merge
    weather_cell = station_time.merge(mapping, on="station_idx", how="inner")
    # explode cell list
    weather_cell = weather_cell.explode("cell_id").reset_index(drop=True)
    # reorder
    cols_keep = ["cell_id","datetime_bin"] + [c for c in weather_cell.columns if c not in ("station_idx","cell_id","datetime_bin")]
    weather_cell = weather_cell[cols_keep]

    out_weather_parquet.parent.mkdir(parents=True, exist_ok=True)
    weather_cell.to_parquet(out_weather_parquet, compression="snappy")
    print("Saved remapped weather to:", out_weather_parquet)
    return weather_cell, grid_cent

# -------------------
# Step B: restrict fire to panel cells
# -------------------
def restrict_fire(panel_path:Path, fire_parquet:Path, out_fire_parquet:Path):
    print("STEP B: Restricting fire rows to panel-active cells.")
    panel = pd.read_parquet(panel_path)
    fire = pd.read_parquet(fire_parquet)
    active_cells = set(panel["cell_id"].unique())
    print("Panel-active cells:", len(active_cells))
    before = len(fire)
    fire = fire[fire["cell_id"].isin(active_cells)].copy()
    after = len(fire)
    print(f"Dropped {before-after} fire rows (outside panel). Kept {after}.")
    # sanitize columns
    if "fire_count" in fire.columns:
        fire["fire_count"] = fire["fire_count"].fillna(0).astype(int)
    else:
        fire["fire_count"] = 0
    if "frp_sum" in fire.columns:
        fire["frp_sum"] = fire["frp_sum"].fillna(0.0)
    else:
        fire["frp_sum"] = 0.0
    if "fire_any" in fire.columns:
        fire["fire_any"] = fire["fire_any"].fillna(False).astype(bool)
    else:
        fire["fire_any"] = fire["fire_count"] > 0
    out_fire_parquet.parent.mkdir(parents=True, exist_ok=True)
    fire.to_parquet(out_fire_parquet, compression="snappy")
    print("Saved filtered fire to:", out_fire_parquet)
    return fire

# -------------------
# Step C: impute static features using kNN
# -------------------
def impute_static(static_parquet:Path, grid_gpkg:Path, panel_cells:set, k:int=8, out_static_parquet:Path=None):
    print("STEP C: Imputing static topo features using k-NN (k=%d)." % k)
    static = pd.read_parquet(static_parquet)
    grid = gpd.read_file(grid_gpkg)
    # ensure centroids exist
    if "centroid_x" not in grid.columns or "centroid_y" not in grid.columns:
        grid["centroid"] = grid.geometry.centroid
        grid["centroid_x"] = grid.centroid.x
        grid["centroid_y"] = grid.centroid.y

    # numeric and categorical cols
    numeric_cols = [c for c in static.select_dtypes(include=[np.number]).columns if c not in ("cell_id",)]
    cat_cols = [c for c in static.select_dtypes(exclude=[np.number]).columns if c not in ("cell_id","geometry","centroid")]

    # prepare static_sources: static rows with centroid coords
    static_sources = static.merge(grid[["cell_id","centroid_x","centroid_y"]], on="cell_id", how="left")
    static_sources = static_sources.dropna(subset=["centroid_x","centroid_y"])
    if static_sources.empty:
        raise RuntimeError("No static source rows with coords found.")

    # build NearestNeighbors on static_sources coords
    X = static_sources[["centroid_x","centroid_y"]].values
    nbrs = NearestNeighbors(n_neighbors=min(k, len(static_sources)), algorithm="ball_tree").fit(X)

    # 1) Impute NaNs inside existing static rows by neighbor stats
    print("Imputing NaNs inside existing static rows...")
    # we'll build a list of imputed rows
    imputed_list = []
    for i, row in static_sources.iterrows():
        cid = row["cell_id"]
        coord = np.array([[row["centroid_x"], row["centroid_y"]]])
        dists, inds = nbrs.kneighbors(coord)
        neighbor_rows = static_sources.iloc[inds[0]].reset_index(drop=True)
        imputed = row.copy()
        for nc in numeric_cols:
            if pd.isna(imputed.get(nc, np.nan)):
                vals = neighbor_rows[nc].dropna()
                imputed[nc] = vals.mean() if len(vals)>0 else np.nan
        for cc in cat_cols:
            if cc in imputed.index and pd.isna(imputed.get(cc, np.nan)):
                vals = neighbor_rows[cc].dropna()
                imputed[cc] = vals.mode().iloc[0] if len(vals)>0 else None
        imputed_list.append(imputed)
    imputed_df = pd.DataFrame(imputed_list)

    # 2) Handle panel cells missing entirely in static: create rows by kNN aggregation
    existing_static_ids = set(static["cell_id"].unique())
    missing_panel_cells = [c for c in panel_cells if c not in existing_static_ids]
    print("Panel cells missing in static:", len(missing_panel_cells))
    new_rows = []
    for cid in missing_panel_cells:
        row_grid = grid[grid["cell_id"]==cid]
        if row_grid.empty:
            continue
        coord = row_grid[["centroid_x","centroid_y"]].values
        dists, inds = nbrs.kneighbors(coord)
        neighbor_rows = static_sources.iloc[inds[0]].reset_index(drop=True)
        new_row = {"cell_id": cid}
        for nc in numeric_cols:
            vals = neighbor_rows[nc].dropna()
            new_row[nc] = vals.mean() if len(vals)>0 else np.nan
        for cc in cat_cols:
            vals = neighbor_rows[cc].dropna()
            new_row[cc] = vals.mode().iloc[0] if len(vals)>0 else None
        new_row["centroid_x"] = coord[0][0]
        new_row["centroid_y"] = coord[0][1]
        new_rows.append(new_row)
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        imputed_df = pd.concat([imputed_df, new_df], ignore_index=True)

    # Merge imputed_df back into original static (fill only missing values)
    static_updated = static.set_index("cell_id")
    imputed_df = imputed_df.set_index("cell_id")
    for col in imputed_df.columns:
        if col in static_updated.columns:
            static_updated[col] = static_updated[col].fillna(imputed_df[col])
        else:
            static_updated[col] = imputed_df[col]

    # final global fills for numeric columns
    for nc in numeric_cols:
        if nc in static_updated.columns:
            static_updated[nc] = static_updated[nc].fillna(static_updated[nc].mean())

    static_updated = static_updated.reset_index()
    out_static = out_static_parquet if (out_static_parquet := Path(static_parquet)) else static_parquet
    static_updated.to_parquet(static_parquet, compression="snappy")
    print("Saved imputed static to:", static_parquet)
    return static_updated

# -------------------
# Finalize: merge cleaned fire into panel
# -------------------
def update_panel_fire(panel_parquet:Path, fire_parquet:Path, out_panel_parquet:Path):
    print("Merging cleaned fire info into panel...")
    panel = pd.read_parquet(panel_parquet)
    fire = pd.read_parquet(fire_parquet)
    # drop existing fire columns
    drop_cols = [c for c in panel.columns if c.startswith("frp") or c.startswith("fire")]
    panel = panel.drop(columns=drop_cols, errors="ignore")
    panel = panel.merge(fire[["cell_id","datetime_bin","fire_count","fire_any","frp_sum"]], on=["cell_id","datetime_bin"], how="left")
    panel["fire_count"] = panel["fire_count"].fillna(0).astype(int)
    panel["fire_any"] = panel["fire_any"].fillna(False).astype(bool)
    panel["frp_sum"] = panel["frp_sum"].fillna(0.0)
    out_panel_parquet.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_panel_parquet, compression="snappy")
    print("Saved updated panel:", out_panel_parquet)
    return panel

# -------------------
# CLI / orchestrator
# -------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agg_dir", required=True)
    p.add_argument("--preproc_dir", required=True)
    p.add_argument("--weather_csv", required=True)
    p.add_argument("--time_res_minutes", type=int, default=60)
    p.add_argument("--k", type=int, default=8)
    args = p.parse_args()

    agg_dir = Path(args.agg_dir)
    preproc_dir = Path(args.preproc_dir)
    weather_csv = Path(args.weather_csv)

    panel_p = agg_dir / "panel_cell_time.parquet"
    weather_out = agg_dir / "weather_cell_time.parquet"
    fire_p = agg_dir / "fire_cell_time.parquet"
    static_p = agg_dir / "cells_static.parquet"

    grid_centroids_csv = preproc_dir / "grid_centroids.csv"
    grid_gpkg = preproc_dir / "grid.gpkg"

    # Step A
    weather_cell, grid_cent = remap_weather(grid_centroids_csv, weather_csv, weather_out, time_res_minutes=args.time_res_minutes)

    # Step B
    fire_clean = restrict_fire(panel_p, fire_p, fire_p)

    # Step C
    panel_df = pd.read_parquet(panel_p)
    panel_cells = set(panel_df["cell_id"].unique())
    static_updated = impute_static(static_p, grid_gpkg, panel_cells, k=args.k, out_static_parquet=static_p)

    # Final: update panel's fire columns
    panel_updated = update_panel_fire(panel_p, fire_p, panel_p)

    print("Post-processing complete. Outputs updated in:", agg_dir)

if __name__ == "__main__":
    main()
