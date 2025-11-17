#!/usr/bin/env python3
"""
baseline_viz_spatial.py

Loads model and feature list JSON (xgb_feature_list.json) and aligns features before prediction.
Produces spatial plot for chosen time_epoch.

Usage same as before.

> python baseline_viz_spatial.py   --grid ./preproc_output/grid.gpkg   --panel ./agg_output/panel_cell_time.parquet   
--edge ./edge_spread_examples_full.parquet   --model ./xgb_edge_baseline.joblib   --time_epoch 175000   --horizon 12   --prob_thresh 0.5
                                      
"""
import argparse, joblib, json
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
import xgboost as xgb

def epoch_hours_from_dt(s):
    return (pd.to_datetime(s).astype("int64") // 10**9) // 3600

def load_inputs(grid_path, panel_path, edge_full_path):
    grid = gpd.read_file(grid_path)
    panel = pd.read_parquet(panel_path)
    if "datetime_bin" in panel.columns:
        panel["datetime_bin"] = pd.to_datetime(panel["datetime_bin"])
        panel["epoch_hr"] = epoch_hours_from_dt(panel["datetime_bin"])
    edge_full = pd.read_parquet(edge_full_path)
    return grid, panel, edge_full

def choose_time(panel, time_epoch=None):
    burning_times = sorted(panel[panel["fire_any"]==True]["epoch_hr"].unique())
    if len(burning_times)==0:
        raise RuntimeError("No burning times found.")
    if time_epoch is None:
        return int(burning_times[len(burning_times)//2])
    if time_epoch not in burning_times:
        arr = np.array(burning_times)
        idx = (np.abs(arr-time_epoch)).argmin()
        return int(arr[idx])
    return int(time_epoch)

def align_features_and_predict(model, df_edge_time, feat_json_path):
    feat_json_path = Path(feat_json_path)
    if feat_json_path.exists():
        feat_cols = json.loads(feat_json_path.read_text())
    else:
        # try to read model metadata
        feat_cols = getattr(model, "feature_names_in_", None)
        if feat_cols is None:
            try:
                feat_cols = model.get_booster().feature_names
            except Exception:
                feat_cols = None
        if feat_cols is None:
            raise RuntimeError("No feature list available; retrain with feature json.")
        feat_cols = list(feat_cols)

    # create missing features with zeros
    for fn in feat_cols:
        if fn not in df_edge_time.columns:
            df_edge_time[fn] = 0.0
    X = df_edge_time[feat_cols].fillna(0.0).values
    # predict probabilities (handle cases where model is xgb sklearn wrapper or booster)
    try:
        probs = model.predict_proba(X)[:,1]
    except Exception:
        # model might be a wrapper that stores a booster from xgb.train
        try:
            bst = model.get_booster()
            dmat = xgb.DMatrix(X, feature_names=feat_cols)
            probs = bst.predict(dmat)
        except Exception as e:
            raise RuntimeError("Model prediction failed: " + str(e))
    df_edge_time = df_edge_time.copy()
    df_edge_time["pred_proba"] = probs
    agg = df_edge_time.groupby("target")["pred_proba"].max().reset_index().rename(columns={"pred_proba":"pred_prob_max"})
    return df_edge_time, agg

def plot_spatial(grid, panel, edge_time_pred, pred_agg, time_epoch, horizon, prob_thresh, out_dir=Path("./viz_output")):
    out_dir.mkdir(parents=True, exist_ok=True)
    # sources at t
    sources = panel[(panel["epoch_hr"]==time_epoch) & (panel["fire_any"]==True)][["cell_id"]].drop_duplicates().rename(columns={"cell_id":"cell_id"})
    centroids = grid[["cell_id","geometry"]].copy()
    centroids["cx"] = centroids.geometry.centroid.x
    centroids["cy"] = centroids.geometry.centroid.y
    sources = sources.merge(centroids[["cell_id","cx","cy"]], left_on="cell_id", right_on="cell_id", how="left")
    # actual new burns in (t, t+H]
    fire_times = panel[panel["fire_any"]==True][["cell_id","epoch_hr"]].drop_duplicates()
    actual_list = []
    lo, hi = time_epoch+1, time_epoch+horizon
    for cid, grp in fire_times.groupby("cell_id"):
        arr = np.sort(grp["epoch_hr"].values)
        idx = np.searchsorted(arr, lo, side='left')
        if idx < len(arr) and arr[idx] <= hi:
            frp_sum = panel[(panel["cell_id"]==cid) & (panel["epoch_hr"]>time_epoch) & (panel["epoch_hr"]<=hi)]["frp_sum"].sum()
            actual_list.append({"target": cid, "future_frp_sum": float(frp_sum)})
    actual_df = pd.DataFrame(actual_list)
    # pred merge with centroids
    pred_merge = centroids.merge(pred_agg, left_on="cell_id", right_on="target", how="left").fillna(0.0)
    fig, axes = plt.subplots(1,2, figsize=(18,10), sharey=True)
    ax1, ax2 = axes
    grid.plot(ax=ax1, color="#f5f5f5", edgecolor="#eee", linewidth=0.2)
    if not actual_df.empty:
        act = centroids.merge(actual_df, left_on="cell_id", right_on="target", how="inner")
        ax1.scatter(act["cx"], act["cy"], s=(act["future_frp_sum"].fillna(1)+1)*10, c="red", alpha=0.9, label="Actual new burn")
    if not sources.empty:
        ax1.scatter(sources["cx"], sources["cy"], s=80, marker="*", color="k", label="Sources (t)")
    ax1.set_title(f"Actual new burns (t -> t+{horizon}h) time_epoch={time_epoch}")
    ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")
    grid.plot(ax=ax2, color="#f5f5f5", edgecolor="#eee", linewidth=0.2)
    sc = ax2.scatter(pred_merge["cx"], pred_merge["cy"], c=pred_merge["pred_prob_max"], cmap="magma", s=18, vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04).set_label("Predicted spread prob (max over sources)")
    top_pred = pred_merge[pred_merge["pred_prob_max"]>=prob_thresh]
    if not top_pred.empty:
        ax2.scatter(top_pred["cx"], top_pred["cy"], s=80, facecolors="none", edgecolors="purple", linewidths=1.5, label=f"Pred >= {prob_thresh:.2f}")
    if not sources.empty:
        ax2.scatter(sources["cx"], sources["cy"], s=80, marker="*", color="k", label="Sources (t)")
    # draw arrows for predicted edges above threshold and actual edges
    pred_edges = edge_time_pred[edge_time_pred["pred_proba"]>=prob_thresh]
    actual_edges = edge_time_pred[edge_time_pred["label"]==1]
    def draw_arrows(df_edges, ax, color="purple"):
        for _, r in df_edges.iterrows():
            sx, sy, tx, ty = r.get("sx"), r.get("sy"), r.get("tx"), r.get("ty")
            if pd.isna(sx) or pd.isna(tx): continue
            ax.annotate("", xy=(tx,ty), xytext=(sx,sy), arrowprops=dict(arrowstyle="->", color=color, alpha=0.6, linewidth=0.8))
    draw_arrows(pred_edges, ax2, color="purple")
    draw_arrows(actual_edges, ax1, color="red")
    ax2.set_title(f"Predicted spread prob (time_epoch={time_epoch})")
    ax1.legend(loc="upper right"); ax2.legend(loc="upper right")
    out_png = Path("./viz_output") / f"spatial_viz_time_{time_epoch}.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print("[saved]", out_png)
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--edge", required=True, help="use full (unbalanced) edge_spread_examples_full.parquet for visualization")
    p.add_argument("--model", required=True)
    p.add_argument("--time_epoch", type=int, default=None)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--prob_thresh", type=float, default=0.5)
    args = p.parse_args()

    grid, panel, edge_full = load_inputs(args.grid, args.panel, args.edge)
    time_epoch = choose_time(panel, args.time_epoch)
    print("[info] chosen time_epoch:", time_epoch)
    edge_time = edge_full[edge_full["time_epoch"]==time_epoch].copy()
    if edge_time.empty:
        raise RuntimeError("No edge examples for the chosen time_epoch.")
    model = joblib.load(args.model)
    feat_json = Path(args.model).with_name("xgb_feature_list.json")
    edge_time_pred, pred_agg = align_features_and_predict(model, edge_time, feat_json)
    plot_spatial(grid, panel, edge_time_pred, pred_agg, time_epoch, args.horizon, args.prob_thresh)