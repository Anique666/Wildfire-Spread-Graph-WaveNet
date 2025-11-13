#!/usr/bin/env python3
"""
data_preprocessing.py

Usage:
    python data_preprocessing.py \
        --topo /mnt/data/topo_data_cleaned.csv \
        --fire /mnt/data/fire_data.csv \
        --weather /mnt/data/output_final_temp_celsius_fixed.csv \
        --out_dir /mnt/data/preproc_output \
        --cell_size_m 1000 \
        --time_col_weather date,time  # if weather uses separate date/time columns (see notes)

Produces:
 - grid.gpkg (grid polygons in metric CRS)
 - topo_with_cell.parquet (topo points with cell index)
 - weather_with_cell.parquet (weather points with cell index)
 - fire_with_cell.parquet (fire points with cell index)
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# geopandas/shapely imports
import geopandas as gpd
from shapely.geometry import box, Point

def estimate_and_project(gdf):
    """Estimate suitable UTM/equal-area CRS and reproject GeoDataFrame. Returns projected gdf and crs."""
    utm = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm)
    return gdf_utm, utm

def build_grid(gdf_utm, cell_size=1000):
    minx, miny, maxx, maxy = gdf_utm.total_bounds
    # ensure coverage to include last point
    xs = np.arange(minx, maxx + cell_size, cell_size)
    ys = np.arange(miny, maxy + cell_size, cell_size)
    polys = []
    ids = []
    for i, x in enumerate(xs[:-1]):
        for j, y in enumerate(ys[:-1]):
            polys.append(box(x, y, x + cell_size, y + cell_size))
            ids.append(f"cell_{i}_{j}")
    grid = gpd.GeoDataFrame({"cell_id": ids, "geometry": polys}, crs=gdf_utm.crs)
    # add centroid coords for easy export
    centroids = grid.geometry.centroid
    grid["centroid_x"] = centroids.x
    grid["centroid_y"] = centroids.y
    return grid

def spatial_join_points_to_grid(points_gdf, grid_gdf, how="left"):
    # ensure same CRS
    if points_gdf.crs != grid_gdf.crs:
        points_gdf = points_gdf.to_crs(grid_gdf.crs)
    # build spatial index for performance
    joined = gpd.sjoin(points_gdf, grid_gdf[["cell_id", "geometry"]], how=how, predicate="within")
    # sjoin adds index_right; rename to cell_id if present
    if "cell_id" in joined.columns:
        return joined
    return joined

def load_points_as_gdf(path, lon_col="longitude", lat_col="latitude", crs_epsg=4326, time_cols=None, parse_dates=None):
    df = pd.read_csv(path)
    # allow alternate column names
    if lon_col not in df.columns or lat_col not in df.columns:
        # try common names
        candidates_lon = [c for c in df.columns if c.lower() in ("lon", "longitude", "long")]
        candidates_lat = [c for c in df.columns if c.lower() in ("lat", "latitude")]
        if not candidates_lon or not candidates_lat:
            raise ValueError(f"Could not find lon/lat columns in {path}")
        lon_col = candidates_lon[0]
        lat_col = candidates_lat[0]
    # create geometry, handle missing lat/lon
    df = df.dropna(subset=[lon_col, lat_col])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=f"EPSG:{crs_epsg}")
    # optional parse combined date/time if user provided
    if time_cols:
        if isinstance(time_cols, (list, tuple)) and len(time_cols) == 2:
            dcol, tcol = time_cols
            if dcol in df.columns and tcol in df.columns:
                gdf["datetime"] = pd.to_datetime(df[dcol].astype(str) + " " + df[tcol].astype(str), errors="coerce")
        else:
            col = time_cols
            if col in df.columns:
                gdf["datetime"] = pd.to_datetime(df[col], errors="coerce")
    if parse_dates:
        for c in parse_dates:
            if c in gdf.columns:
                gdf[c] = pd.to_datetime(gdf[c], errors="coerce")
    return gdf

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading topo points...")
    topo_gdf = load_points_as_gdf(args.topo, lon_col=args.topo_lon, lat_col=args.topo_lat)
    print(f"Topo points: {len(topo_gdf)}")

    print("Loading weather points...")
    # weather may have separate date and time columns
    weather_time_cols = None
    if args.weather_time_cols:
        weather_time_cols = args.weather_time_cols.split(",")
    weather_gdf = load_points_as_gdf(args.weather, lon_col=args.weather_lon, lat_col=args.weather_lat, time_cols=weather_time_cols)
    print(f"Weather rows: {len(weather_gdf)}")

    print("Loading fire points...")
    # fire may have date/time columns acq_date/acq_time or similar
    fire_time_cols = None
    if args.fire_time_cols:
        fire_time_cols = args.fire_time_cols.split(",")
    fire_gdf = load_points_as_gdf(args.fire, lon_col=args.fire_lon, lat_col=args.fire_lat, time_cols=fire_time_cols)
    print(f"Fire detections: {len(fire_gdf)}")

    # Combine all points to estimate CRS
    combined = pd.concat([topo_gdf.drop(columns="geometry"), weather_gdf.drop(columns="geometry"), fire_gdf.drop(columns="geometry")], ignore_index=True, sort=False)
    # create temp combined geometry for CRS estimation: use topo alone if combined too big
    sample_gdf = topo_gdf if len(topo_gdf) > 0 else (weather_gdf if len(weather_gdf) > 0 else fire_gdf)
    print("Estimating metric CRS (UTM) from sample...")
    sample_projected, proj_crs = estimate_and_project(sample_gdf)
    print("Projected CRS:", proj_crs)

    # Project each gdf to projected CRS
    topo_utm = topo_gdf.to_crs(proj_crs)
    weather_utm = weather_gdf.to_crs(proj_crs)
    fire_utm = fire_gdf.to_crs(proj_crs)

    print("Building 1x1 km grid...")
    grid = build_grid(topo_utm, cell_size=args.cell_size_m)
    print("Num grid cells:", len(grid))

    # Save grid (projected)
    grid_path = out_dir / "grid.gpkg"
    grid.to_file(grid_path, driver="GPKG")
    print("Saved grid:", grid_path)

    # Spatial join: assign points to grid cells
    print("Spatial-joining topo -> grid...")
    topo_joined = gpd.sjoin(topo_utm, grid[["cell_id", "geometry"]], how="left", predicate="within")
    topo_out = out_dir / "topo_with_cell.parquet"
    topo_joined.to_parquet(topo_out)
    print("Saved:", topo_out)

    print("Spatial-joining weather -> grid...")
    weather_joined = gpd.sjoin(weather_utm, grid[["cell_id", "geometry"]], how="left", predicate="within")
    weather_out = out_dir / "weather_with_cell.parquet"
    weather_joined.to_parquet(weather_out)
    print("Saved:", weather_out)

    print("Spatial-joining fire -> grid...")
    fire_joined = gpd.sjoin(fire_utm, grid[["cell_id", "geometry"]], how="left", predicate="within")
    fire_out = out_dir / "fire_with_cell.parquet"
    fire_joined.to_parquet(fire_out)
    print("Saved:", fire_out)

    # Also save grid centroids in lat/lon for mapping/visualization
    grid_latlon = grid.to_crs("EPSG:4326")
    grid_latlon["lon"] = grid_latlon.centroid.x
    grid_latlon["lat"] = grid_latlon.centroid.y
    grid_latlon_out = out_dir / "grid_centroids.csv"
    grid_latlon[["cell_id", "lon", "lat"]].to_csv(grid_latlon_out, index=False)
    print("Saved grid centroids:", grid_latlon_out)

    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--topo", required=True)
    p.add_argument("--fire", required=True)
    p.add_argument("--weather", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cell_size_m", type=int, default=1000)
    # column names: override if your CSVs use different names
    p.add_argument("--topo_lon", default="longitude")
    p.add_argument("--topo_lat", default="latitude")
    p.add_argument("--weather_lon", default="lon")
    p.add_argument("--weather_lat", default="lat")
    p.add_argument("--fire_lon", default="longitude")
    p.add_argument("--fire_lat", default="latitude")
    # optionally pass comma-separated date,time column names
    p.add_argument("--weather_time_cols", default=None, help="comma-separated date_col,time_col or single datetime col")
    p.add_argument("--fire_time_cols", default=None, help="comma-separated date_col,time_col or single datetime col")
    args = p.parse_args()
    main(args)
