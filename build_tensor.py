#!/usr/bin/env python3
"""
build_dense_tensor.py

Build a dense node x time x features tensor (memmap) and labels for spread prediction.

Outputs (in --out_dir):
 - X.dat           : memmap float32 with shape (N_nodes, T, F_time)
 - y_future.dat    : memmap uint8 with shape (N_nodes, T)  (label: target burns in (t, t+H] and was not burning at t)
 - y_current.dat   : memmap uint8 with shape (N_nodes, T)  (fire_any at t)  -- diagnostic
 - node_static.npy : small float32 array (N_nodes, F_static) with static/topo features
 - meta.json       : metadata (node_list, time_index (iso strings), feature names, shapes, epoch_min/max)
 
Design notes:
 - By default uses only "active nodes" found in panel (unique cell_id in panel_cell_time.parquet).
 - Time axis is integer hours since epoch (epoch_hr) between min and max in panel.
 - Memmaps used to avoid large RAM use.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import math
import sys
from tqdm import tqdm

def epoch_hours_from_dt(s):
    return (pd.to_datetime(s).astype("int64") // 10**9) // 3600

def safe_cast_float32(arr):
    return np.asarray(arr, dtype=np.float32)

def main(panel_parquet, static_parquet, grid_gpkg, out_dir,
         use_all_nodes=False, horizon=24, time_res_hours=1, time_min=None, time_max=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading panel parquet (this may take a moment)...")
    panel = pd.read_parquet(panel_parquet)
    # ensure epoch_hr present
    if "datetime_bin" in panel.columns:
        panel["datetime_bin"] = pd.to_datetime(panel["datetime_bin"])
        panel["epoch_hr"] = epoch_hours_from_dt(panel["datetime_bin"])
    elif "epoch_hr" not in panel.columns:
        raise RuntimeError("panel must contain datetime_bin or epoch_hr column")

    # Time range
    if time_min is None:
        epoch_min = int(panel["epoch_hr"].min())
    else:
        epoch_min = int(pd.to_datetime(time_min).astype("int64") // 10**9 // 3600)
    if time_max is None:
        epoch_max = int(panel["epoch_hr"].max())
    else:
        epoch_max = int(pd.to_datetime(time_max).astype("int64") // 10**9 // 3600)

    print(f"[info] epoch range hours: {epoch_min} -> {epoch_max} (inclusive)")
    # build uniform time index at given resolution
    time_index = np.arange(epoch_min, epoch_max + 1, time_res_hours, dtype=np.int64)
    T = len(time_index)
    print(f"[info] time steps (T): {T}")

    # choose nodes
    if use_all_nodes:
        print("[info] Using all grid nodes from grid.gpkg (this may be large).")
        grid = gpd.read_file(grid_gpkg)
        node_list = list(grid["cell_id"].unique())
    else:
        node_list = sorted(panel["cell_id"].unique().tolist())
        print(f"[info] Using active nodes from panel: N = {len(node_list)}")

    N = len(node_list)

    # time feature columns to attempt to use (present in panel)
    candidate_time_feats = ["frp_sum", "t2m_mean", "wind_speed_mean", "u10_mean", "v10_mean"]
    present_time_feats = [c for c in candidate_time_feats if c in panel.columns]
    print("[info] Time-varying features included:", present_time_feats)
    F_time = len(present_time_feats)

    # load static features (small)
    print("[1.5] Loading static features...")
    static = pd.read_parquet(static_parquet)
    static = static.set_index("cell_id")
    # choose numeric static columns (exclude geometry)
    static_num = [c for c in static.columns if np.issubdtype(static[c].dtype, np.number)]
    # keep up to a reasonable number (you can change)
    static_cols = static_num  # use all numeric static cols present
    print(f"[info] static numeric features included ({len(static_cols)}): {static_cols}")

    F_static = len(static_cols)

    # prepare memmaps
    X_path = out_dir / "X.dat"
    y_future_path = out_dir / "y_future.dat"
    y_current_path = out_dir / "y_current.dat"
    node_static_path = out_dir / "node_static.npy"
    meta_path = out_dir / "meta.json"

    print("[2/6] Creating memmaps (this reserves disk space)...")
    X = np.memmap(str(X_path), dtype=np.float32, mode="w+", shape=(N, T, F_time))
    y_future = np.memmap(str(y_future_path), dtype=np.uint8, mode="w+", shape=(N, T))
    y_current = np.memmap(str(y_current_path), dtype=np.uint8, mode="w+", shape=(N, T))
    node_static = np.zeros((N, F_static), dtype=np.float32)

    # build a quick lookup from panel for present features by (epoch, cell)
    print("[3/6] Building quick lookup dicts for time features and fire flags...")
    panel_small = panel[["epoch_hr", "cell_id"] + present_time_feats + ["fire_any"]].drop_duplicates()
    # convert to numeric and fill NaNs in source table with np.nan (we will fill per-node later)
    panel_small = panel_small.fillna(np.nan)
    # group by cell for faster per-node operations
    grouped = panel_small.groupby("cell_id")

    # mapping from node id to index
    node_to_idx = {nid: i for i, nid in enumerate(node_list)}

    # Create a map from epoch -> time_index position (fast)
    epoch_to_pos = {int(e): idx for idx, e in enumerate(time_index)}

    print("[4/6] Filling memmaps per node (forward/backfill for missing times).")
    # iterate nodes and fill into memmaps (this keeps memory small)
    for i, nid in enumerate(tqdm(node_list, desc="nodes")):
        # initialize per-node arrays
        node_X = np.zeros((T, F_time), dtype=np.float32)
        node_ycur = np.zeros((T,), dtype=np.uint8)

        if nid in grouped.groups:
            sub = grouped.get_group(nid).copy()
            # set index by epoch
            sub_idx = sub.set_index("epoch_hr")
            # assign values for epochs available
            for feat_j, feat in enumerate(present_time_feats):
                # iterate rows and fill
                for epoch, val in sub_idx[feat].items():
                    pos = epoch_to_pos.get(int(epoch))
                    if pos is not None:
                        if val is None or (isinstance(val, float) and np.isnan(val)):
                            node_X[pos, feat_j] = np.nan
                        else:
                            node_X[pos, feat_j] = float(val)
            # fire flag (current)
            if "fire_any" in sub_idx.columns:
                for epoch, val in sub_idx["fire_any"].items():
                    pos = epoch_to_pos.get(int(epoch))
                    if pos is not None:
                        node_ycur[pos] = 1 if bool(val) else 0

            # now simple ffill then bfill along time axis for each feature
            for feat_j in range(F_time):
                col = node_X[:, feat_j]
                # mask NaN -> use np.nan for fill
                mask = np.isnan(col)
                if mask.all():
                    # no data for this feature for this node -> leave zeros
                    col[:] = 0.0
                else:
                    # forward fill
                    # find indices with valid entries
                    valid_idx = np.where(~mask)[0]
                    # fill between valid points
                    # ffill
                    last = None
                    for k in range(T):
                        if not math.isnan(col[k]):
                            last = col[k]
                        else:
                            if last is not None:
                                col[k] = last
                    # bfill for leading NaNs
                    if math.isnan(col[0]):
                        next_valid = valid_idx[0]
                        col[:next_valid] = col[next_valid]
                    # remaining NaNs (if any) -> set to 0
                    col[np.isnan(col)] = 0.0
                    node_X[:, feat_j] = col
        else:
            # node has no rows in panel => keep zeros
            pass

        # write into memmaps
        X[i, :, :] = node_X
        y_current[i, :] = node_ycur

        # node static features
        if nid in static.index:
            vals = static.loc[nid, static_cols].astype(np.float32).values
            # replace NaN with 0
            vals = np.nan_to_num(vals, nan=0.0)
            node_static[i, :] = vals
        else:
            node_static[i, :] = np.zeros((F_static,), dtype=np.float32)

    # compute y_future label: for each node/time, check if node becomes burning in (t, t+H] and was not burning at t
    print("[5/6] Building y_future labels with horizon (hours) =", horizon)
    # first build per-node arrays of fire times from y_current memmap
    for i in tqdm(range(N), desc="labeling"):
        # use vectorized rolling lookup: for each t check if any y_current[t+1:t+H] == 1
        cur = y_current[i, :].astype(np.uint8)
        # convolution trick: sliding window sum
        if horizon >= 1:
            # create cumulative sum
            csum = np.concatenate(([0], np.cumsum(cur, dtype=np.int32)))
            # for position t, sum in (t, t+H] is csum[t+H] - csum[t]
            # need to handle beyond array end
            for tpos in range(T):
                lo = tpos + 1
                hi = min(tpos + horizon, T - 1)
                if lo > hi:
                    y_future[i, tpos] = 0
                else:
                    s = int(csum[hi + 1] - csum[lo])
                    # positive only if not burning at tpos (cur[tpos]==0) AND s>0
                    y_future[i, tpos] = 1 if (cur[tpos] == 0 and s > 0) else 0
        else:
            y_future[i, :] = 0

    # save node static and metadata
    print("[6/6] Saving node_static and metadata")
    np.save(str(node_static_path), node_static)
    meta = {
        "node_list": node_list,
        "time_index_epoch_hr": time_index.tolist(),
        "time_index_iso": [pd.to_datetime(int(e)*3600, unit="s").isoformat() for e in time_index],
        "present_time_feats": present_time_feats,
        "static_features": static_cols,
        "shapes": {
            "X_shape": [N, T, F_time],
            "y_future_shape": [N, T],
            "node_static_shape": [N, F_static]
        },
        "epoch_min": int(epoch_min),
        "epoch_max": int(epoch_max),
        "horizon_hours": int(horizon)
    }
    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)

    # flush memmaps to disk
    X.flush()
    y_future.flush()
    y_current.flush()
    print("[done] Files written to:", out_dir)
    print(" - X (memmap):", X_path)
    print(" - y_future (memmap):", y_future_path)
    print(" - y_current (memmap):", y_current_path)
    print(" - node_static (npy):", node_static_path)
    print(" - meta:", meta_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True, help="agg_output/panel_cell_time.parquet")
    p.add_argument("--static", required=True, help="agg_output/cells_static.parquet")
    p.add_argument("--grid", required=True, help="preproc_output/grid.gpkg (for node_list if use_all_nodes)")
    p.add_argument("--out_dir", default="./tensor_output")
    p.add_argument("--use_all_nodes", action="store_true", help="If set, include all grid cells instead of only active nodes (may be very large)")
    p.add_argument("--horizon", type=int, default=24, help="Horizon hours for y_future labels")
    p.add_argument("--time_res_hours", type=int, default=1, help="Time resolution in hours")
    p.add_argument("--time_min", default=None, help="Optional ISO time string to force start time")
    p.add_argument("--time_max", default=None, help="Optional ISO time string to force end time")
    args = p.parse_args()
    main(args.panel, args.static, args.grid, args.out_dir, use_all_nodes=args.use_all_nodes, horizon=args.horizon, time_res_hours=args.time_res_hours, time_min=args.time_min, time_max=args.time_max)
