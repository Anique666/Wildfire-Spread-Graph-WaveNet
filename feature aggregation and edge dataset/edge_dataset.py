#!/usr/bin/env python3
"""
build_edge_dataset.py

Build directed edge examples and save:
 - edge_spread_examples_full.parquet (unbalanced)
 - edge_spread_examples.parquet (balanced)
 - edge_feature_list.json (ordered list of numeric feature column names for modeling)

 python edge_dataset.py --panel ./agg_output/panel_cell_time.parquet --grid ./preproc_output/grid.gpkg --static ./agg_output/cells_static.parquet --out ./edge_spread_examples.parquet --k 8 --horizon 12 --lags 6 --neg_ratio 3
"""
import argparse, time, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")

def epoch_hours_from_dt(s):
    return (pd.to_datetime(s).astype("int64") // 10**9) // 3600

def build_neighbor_pairs(grid_gpkg, active_cells, k):
    grid = gpd.read_file(grid_gpkg)
    centroids = grid[grid['cell_id'].isin(active_cells)][['cell_id','centroid_x','centroid_y']].dropna()
    centroids = centroids.set_index('cell_id').loc[active_cells].reset_index()
    coords = centroids[['centroid_x','centroid_y']].values
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(coords)), algorithm='ball_tree').fit(coords)
    dists, inds = nbrs.kneighbors(coords)
    rows = []
    ids = centroids['cell_id'].tolist()
    for i_idx, i_cell in enumerate(ids):
        for n_idx in range(1, dists.shape[1]):  # skip self
            j_cell = ids[inds[i_idx, n_idx]]
            rows.append((i_cell, j_cell, float(dists[i_idx, n_idx])))
    neigh_df = pd.DataFrame(rows, columns=["source","target","dist_m"])
    return neigh_df, centroids

def compute_frp_trend(panel_lookup, source, time_epoch, lags):
    # returns slope (per hour) of frp for last lags hours prior to time_epoch
    xs = []
    ys = []
    for lag in range(1, lags+1):
        tlag = time_epoch - lag
        val = panel_lookup.get((tlag, source), None)
        if val is not None:
            xs.append(-lag)  # relative to t, negative direction
            ys.append(val)
    if len(xs) >= 2:
        lr = LinearRegression()
        lr.fit(np.array(xs).reshape(-1,1), np.array(ys))
        return float(lr.coef_[0])
    else:
        return 0.0

def build_edge_dataset(panel_parquet, grid_gpkg, static_parquet, out_prefix,
                       k=8, horizon=24, lags=6, neg_ratio=8):
    t0 = time.time()
    panel = pd.read_parquet(panel_parquet)
    if "datetime_bin" in panel.columns:
        panel["datetime_bin"] = pd.to_datetime(panel["datetime_bin"])
        panel["epoch_hr"] = epoch_hours_from_dt(panel["datetime_bin"])
    elif "epoch_hr" not in panel.columns:
        raise RuntimeError("panel must have datetime_bin or epoch_hr")
    # ensure FRP/fire cols exist
    panel["frp_sum"] = panel.get("frp_sum", 0.0)
    panel["fire_any"] = panel.get("fire_any", False).astype(bool)

    active_cells = sorted(panel["cell_id"].unique())
    print("[info] active cells:", len(active_cells))

    neigh_df, centroids = build_neighbor_pairs(grid_gpkg, active_cells, k)
    print("[info] neighbor pairs:", len(neigh_df))

    # prepare lookup maps for fast access
    # panel_lookup: dict[(epoch_hr, cell_id)] -> frp_sum or fire_any
    print("[info] building lookup structures...")
    panel_small = panel[["epoch_hr","cell_id","frp_sum","fire_any"]].drop_duplicates()
    panel_lookup_frp = {(int(r.epoch_hr), r.cell_id): float(r.frp_sum) for r in panel_small.itertuples()}
    panel_lookup_fire = {(int(r.epoch_hr), r.cell_id): bool(r.fire_any) for r in panel_small.itertuples()}

    # times when any source is burning (we create examples only when source is burning)
    burning = panel[panel["fire_any"]==True][["epoch_hr","cell_id"]].drop_duplicates().rename(columns={"cell_id":"source","epoch_hr":"time_epoch"})
    print("[info] burning events (unique):", len(burning))

    # cross-join burning x neighbors
    examples = burning.merge(neigh_df, on="source", how="inner")
    print("[info] raw examples (burning x neighbors):", len(examples))

    # merge source t features and target t features (vectorized)
    # First, convert panel_small into DataFrame keyed for merges
    panel_for_merge = panel_small.rename(columns={"epoch_hr":"time_epoch"})
    src_panel = panel_for_merge.rename(columns={"cell_id":"source","frp_sum":"frp_src","fire_any":"fire_any_src"})
    tgt_panel = panel_for_merge.rename(columns={"cell_id":"target","frp_sum":"frp_tgt","fire_any":"fire_any_tgt"})
    examples = examples.merge(src_panel, on=["time_epoch","source"], how="left")
    examples = examples.merge(tgt_panel, on=["time_epoch","target"], how="left")

    # ensure target was NOT burning at time t (we only want spread)
    examples["target_burning_at_t"] = examples["fire_any_tgt"].fillna(False)
    examples = examples[~examples["target_burning_at_t"].astype(bool)].copy()  # drop cases where target already burning (not spread)

    # compute future label: target burns in (t, t+H]
    print("[info] labeling (first-ignition in horizon=%d hours)..." % horizon)
    # build sorted arrays of fire times per target for faster search
    fire_times = panel[panel["fire_any"]==True][["cell_id","epoch_hr"]].drop_duplicates().sort_values(["cell_id","epoch_hr"])
    fire_dict = {cid: grp["epoch_hr"].values for cid, grp in fire_times.groupby("cell_id")}
    def label_row(t_epoch, tgt):
        arr = fire_dict.get(tgt)
        if arr is None or len(arr)==0:
            return 0
        lo, hi = int(t_epoch) + 1, int(t_epoch) + horizon
        idx = np.searchsorted(arr, lo, side='left')
        return 1 if idx < len(arr) and arr[idx] <= hi else 0
    examples["label"] = examples.apply(lambda r: label_row(r["time_epoch"], r["target"]), axis=1)

    # compute neighbor aggregates at time t for the source: burning_neighbor_count, mean_frp_neighbors
    print("[info] computing neighbor aggregates...")
    # build list of neighbors per source for fast lookup
    neigh_map = neigh_df.groupby("source")["target"].apply(list).to_dict()

    def neighbor_stats(time_epoch, source):
        neighs = neigh_map.get(source, [])
        bcount = 0
        frp_vals = []
        for n in neighs:
            val = panel_lookup_frp.get((int(time_epoch), n))
            fireflag = panel_lookup_fire.get((int(time_epoch), n), False)
            if fireflag:
                bcount += 1
            if val is not None:
                frp_vals.append(val)
        mean_frp = float(np.mean(frp_vals)) if frp_vals else 0.0
        return bcount, mean_frp

    # vectorized-ish apply (we'll compute in loop but it's over examples length; acceptable)
    bcounts = []
    mean_frps = []
    frp_trends = []
    print("[info] computing per-example neighbor counts and trends (this may take a few minutes)...")
    # precompute a simple quick lookup for FRP at (epoch,cell)
    # compute frp trend per example using compute_frp_trend helper
    for r in examples.itertuples(index=False):
        te = int(r.time_epoch)
        src = r.source
        bcount, mean_frp = neighbor_stats(te, src)
        bcounts.append(bcount)
        mean_frps.append(mean_frp)
        trend = compute_frp_trend(panel_lookup_frp, src, te, lags)
        frp_trends.append(trend)
    examples["burning_neighbor_count"] = bcounts
    examples["mean_frp_neighbors"] = mean_frps
    examples["frp_src_trend"] = frp_trends

    # geometry: attach centroids and compute vector angle, wind alignment if wind present
    print("[info] adding geometry & wind alignment features...")
    grid = gpd.read_file(grid_gpkg)[["cell_id","centroid_x","centroid_y"]]
    cent_map = grid.set_index("cell_id")[["centroid_x","centroid_y"]]
    examples["sx"] = examples["source"].map(cent_map["centroid_x"])
    examples["sy"] = examples["source"].map(cent_map["centroid_y"])
    examples["tx"] = examples["target"].map(cent_map["centroid_x"])
    examples["ty"] = examples["target"].map(cent_map["centroid_y"])
    examples["vec_dx"] = examples["tx"] - examples["sx"]
    examples["vec_dy"] = examples["ty"] - examples["sy"]
    examples["vec_angle_deg"] = (np.degrees(np.arctan2(examples["vec_dy"], examples["vec_dx"])) + 360) % 360
    if "wind_dir_src" in examples.columns:
        diff = (examples["wind_dir_src"].fillna(0) - examples["vec_angle_deg"] + 180) % 360 - 180
        examples["wind_align"] = np.cos(np.deg2rad(diff))
    else:
        examples["wind_align"] = 0.0

    # attach a few static features if available (small set)
    static = pd.read_parquet(static_parquet)
    static_small = static[["cell_id"] + [c for c in static.columns if c not in ("cell_id","geometry")][:6]].copy()
    src_static = static_small.rename(columns={"cell_id":"source"})
    tgt_static = static_small.rename(columns={"cell_id":"target"})
    examples = examples.merge(src_static, on="source", how="left")
    examples = examples.merge(tgt_static, on="target", how="left", suffixes=("","_tgt"))

    # fill na and choose numeric feature columns in stable order
    examples = examples.fillna(0.0)
    numeric_cols = [c for c in examples.select_dtypes(include=[np.number]).columns if c not in ("label",)]
    # exclude ids/time we don't want as features
    exclude = {"time_epoch","sx","sy","tx","ty"}
    feat_cols = [c for c in numeric_cols if c not in exclude]
    # reorder features: put key features first for readability
    preferred_order = ["dist_m","frp_src","frp_src_trend","burning_neighbor_count","mean_frp_neighbors","wind_align"]
    feat_final = [c for c in preferred_order if c in feat_cols] + [c for c in feat_cols if c not in preferred_order]
    # save JSON
    feat_json_path = Path(out_prefix).with_name("edge_feature_list.json")
    feat_json_path.write_text(json.dumps(feat_final))
    print("[info] saved feature list:", feat_json_path)

    # balance: keep all positives, sample negatives neg_ratio * positives
    pos = examples[examples["label"]==1]
    neg = examples[examples["label"]==0]
    n_pos = len(pos)
    n_neg_keep = int(min(len(neg), neg_ratio * max(1, n_pos)))
    neg_sample = neg.sample(n=n_neg_keep, random_state=42) if n_neg_keep>0 else neg
    balanced = pd.concat([pos, neg_sample]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # save outputs
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    full_path = out_prefix.with_name(out_prefix.stem + "_full.parquet")
    balanced_path = out_prefix
    examples.to_parquet(full_path, compression="snappy")
    balanced.to_parquet(balanced_path, compression="snappy")
    print("[done] full saved:", full_path)
    print("[done] balanced saved:", balanced_path)
    print("[time] seconds:", time.time()-t0)
    return balanced_path, full_path, feat_json_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--static", required=True)
    p.add_argument("--out", default="./edge_spread_examples.parquet")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--lags", type=int, default=6)
    p.add_argument("--neg_ratio", type=float, default=8.0)
    args = p.parse_args()
    build_edge_dataset(args.panel, args.grid, args.static, args.out, k=args.k, horizon=args.horizon, lags=args.lags, neg_ratio=args.neg_ratio)