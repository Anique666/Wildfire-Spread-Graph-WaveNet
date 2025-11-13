#!/usr/bin/env python3
"""
build_adjacency.py

Builds a k-NN graph adjacency for your node_list (from meta.json) using grid centroids.
Options:
 - --meta     : path to meta.json (to get node_list)
 - --grid     : path to grid.gpkg (must contain cell_id, centroid_x, centroid_y)
 - --out_dir  : output dir (saves adj.npy, adj_csr.npz, edge_index.npy, edge_weight.npy, adj_meta.json)
 - --k        : neighbors (default 8)
 - --use_wind : include wind reweighting based on mean u10/v10 from panel (default: False)
 - --wind_strength : amount to upweight downwind edges in [0..1], default 0.6
 - --normalize : 'rw' (row-normalize) or 'sym' (symmetric) or 'none'
 - --sparse_only : save only sparse outputs (avoid writing dense adj.npy)

Example:
python build_adjacency.py --meta ./tensors_full/meta.json --grid ./preproc_output/grid.gpkg --out_dir ./adj_output --k 8 --use_wind --wind_strength 0.6 --normalize rw
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
import math
import pandas as pd

def load_meta(meta_path):
    with open(meta_path,"r") as f:
        return json.load(f)

def read_centroids(grid_path, node_list):
    g = gpd.read_file(grid_path)
    # ensure centroid_x/y exist; if geometry present compute centroid coords
    if "centroid_x" not in g.columns or "centroid_y" not in g.columns:
        g["centroid_x"] = g.geometry.centroid.x
        g["centroid_y"] = g.geometry.centroid.y
    cent = g[["cell_id","centroid_x","centroid_y"]].set_index("cell_id")
    # keep only nodes in node_list and preserve ordering
    coords = []
    missing = []
    for n in node_list:
        if n in cent.index:
            coords.append((cent.loc[n,"centroid_x"], cent.loc[n,"centroid_y"]))
        else:
            coords.append((np.nan,np.nan))
            missing.append(n)
    if missing:
        print("[warn] missing centroids for nodes:", len(missing))
    coords = np.array(coords, dtype=np.float64)
    return coords

def compute_wind_dir_from_panel(panel_parquet, node_list):
    # compute mean u10,v10 per node across time (if available)
    df = pd.read_parquet(panel_parquet, columns=["cell_id","u10_mean","v10_mean"])
    df = df.groupby("cell_id").agg({"u10_mean":"mean","v10_mean":"mean"}).reset_index()
    df = df.set_index("cell_id")
    wind_map = {}
    for n in node_list:
        if n in df.index:
            u = df.loc[n,"u10_mean"]
            v = df.loc[n,"v10_mean"]
            if pd.isna(u) or pd.isna(v):
                wind_map[n] = None
            else:
                # meteorological wind direction degrees (from where wind comes)
                # convert vector to direction degrees (0..360)
                dir_deg = (math.degrees(math.atan2(v, u)) + 360) % 360
                wind_map[n] = {"u":float(u),"v":float(v),"dir_deg":float(dir_deg)}
        else:
            wind_map[n] = None
    return wind_map

def build_knn_edges(coords, k):
    # coords shape (N,2)
    valid = ~np.isnan(coords).any(axis=1)
    idx_map = np.arange(len(coords))
    nbrs = NearestNeighbors(n_neighbors=min(k+1,int(valid.sum())), algorithm="ball_tree").fit(coords[valid])
    dists, inds = nbrs.kneighbors(coords[valid])
    # map valid indices back to full index
    full_inds = np.where(valid)[0]
    rows=[]
    for i_local, i_global in enumerate(full_inds):
        for j in range(1, inds.shape[1]):  # skip self at j=0
            j_global = full_inds[inds[i_local,j]]
            rows.append((i_global, j_global, float(dists[i_local,j])))
    return rows

def distance_weight(d, scale):
    # smooth exponential kernel
    if scale <= 0:
        return 1.0 / (1.0 + d)
    return math.exp(-d / scale)

def add_wind_reweight(edges, node_list, wind_map, coords, wind_strength=0.6):
    # edges: list of (i,j,dist)
    # compute vector angle from i->j and compare to wind direction at i
    new_edges=[]
    for i,j,d in edges:
        w = distance_weight(d, scale)
        wd = wind_map.get(node_list[i], None)
        if wd is None:
            new_edges.append((i,j,w))
            continue
        vec = coords[j] - coords[i]
        if np.isnan(vec).any():
            new_edges.append((i,j,w))
            continue
        angle = (math.degrees(math.atan2(vec[1], vec[0])) + 360) % 360
        diff = abs((wd["dir_deg"] - angle + 180) % 360 - 180)  # -180..180
        cosdiff = math.cos(math.radians(diff))
        # align factor: if cosdiff>0 (downwind direction), upweight
        align = max(0.0, cosdiff)
        factor = 1.0 + wind_strength * align
        new_edges.append((i,j, w * factor))
    return new_edges

def normalize_adj_csr(A_csr, method="rw"):
    # A_csr: scipy csr matrix
    if method == "none":
        return A_csr
    if method == "rw":
        # row normalize (D^-1 A)
        rowsum = np.array(A_csr.sum(axis=1)).flatten()
        rowsum[rowsum==0] = 1.0
        inv = 1.0 / rowsum
        Dinv = sp.diags(inv)
        return Dinv.dot(A_csr)
    if method == "sym":
        rowsum = np.array(A_csr.sum(axis=1)).flatten()
        rowsum[rowsum==0] = 1.0
        dinv_sqrt = 1.0 / np.sqrt(rowsum)
        DinvS = sp.diags(dinv_sqrt)
        return DinvS.dot(A_csr).dot(DinvS)
    raise ValueError("Unknown normalize method")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True, help="meta.json produced by build_dense_tensor / tensor meta")
    p.add_argument("--grid", required=True, help="grid.gpkg with centroids")
    p.add_argument("--panel", default=None, help="panel parquet to compute mean wind vectors (optional, recommended if --use_wind)")
    p.add_argument("--out_dir", default="./adj_output")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--use_wind", action="store_true")
    p.add_argument("--wind_strength", type=float, default=0.6)
    p.add_argument("--normalize", choices=["rw","sym","none"], default="rw")
    p.add_argument("--sparse_only", action="store_true")
    args = p.parse_args()

    meta = load_meta(args.meta)
    node_list = meta["node_list"]
    coords = read_centroids(args.grid, node_list)  # (N,2)
    N = len(node_list)
    # compute scale (median distance among kNN raw distances) for kernel
    print("[info] building kNN edges k=", args.k)
    raw_edges = build_knn_edges(coords, args.k)
    if len(raw_edges) == 0:
        raise RuntimeError("No kNN edges found (check coordinates).")
    dists = np.array([d for (_,_,d) in raw_edges])
    scale = float(np.median(dists) if len(dists)>0 else 1.0)
    print("[info] distance scale (median):", scale)

    # attach initial distance-weighted edges
    edges = [(i,j, distance_weight(d, scale)) for (i,j,d) in raw_edges]

    # optional wind reweighting
    if args.use_wind:
        if args.panel is None:
            raise RuntimeError("--use_wind requires --panel to compute mean wind vectors")
        print("[info] computing mean wind per node from panel:", args.panel)
        wind_map = compute_wind_dir_from_panel(args.panel, node_list)
        # we will use the external scale variable inside add_wind_reweight by closure
        # to allow usage here, set global
        # recreate weight function to use same distance scale
        def distance_weight_local(d):
            return math.exp(-d/scale)
        # do wind reweight
        new_edges=[]
        for i,j,d in raw_edges:
            w = distance_weight(d, scale)
            wd = wind_map.get(node_list[i], None)
            if wd is None:
                new_edges.append((i,j,w))
                continue
            vec = coords[j] - coords[i]
            if np.isnan(vec).any():
                new_edges.append((i,j,w))
                continue
            angle = (math.degrees(math.atan2(vec[1], vec[0])) + 360) % 360
            diff = abs((wd["dir_deg"] - angle + 180) % 360 - 180)
            cosdiff = math.cos(math.radians(diff))
            align = max(0.0, cosdiff)
            factor = 1.0 + args.wind_strength * align
            new_edges.append((i,j, w * factor))
        edges = new_edges

    # Build sparse adjacency matrix
    rows = [i for (i,j,w) in edges]
    cols = [j for (i,j,w) in edges]
    data = [w for (i,j,w) in edges]
    A = sp.coo_matrix((data, (rows, cols)), shape=(N,N))
    # make symmetric by adding transpose (optional; we will symmetrize weights)
    A = A.tocsr()
    # ensure graph is undirected-ish: A = A + A.T (this makes edges bidirectional)
    A = (A + A.transpose()).tocsr()

    print("[info] raw adjacency built: edges (nonzeros):", A.nnz)
    # normalize
    A_norm = normalize_adj_csr(A, method=args.normalize)
    # save outputs
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # sparse
    sp.save_npz(str(out_dir / "adj_csr.npz"), A_norm)
    # save edge_index and edge_weight for PyG
    Acoo = A_norm.tocoo()
    edge_index = np.vstack([Acoo.row.astype(np.int64), Acoo.col.astype(np.int64)])
    edge_weight = Acoo.data.astype(np.float32)
    np.save(str(out_dir / "edge_index.npy"), edge_index)
    np.save(str(out_dir / "edge_weight.npy"), edge_weight)
    # dense (optional)
    if not args.sparse_only:
        adj_dense = A_norm.toarray().astype(np.float32)
        np.save(str(out_dir / "adj.npy"), adj_dense)
    # meta
    meta_out = {
        "node_count": N,
        "k": args.k,
        "normalize": args.normalize,
        "use_wind": bool(args.use_wind),
        "wind_strength": float(args.wind_strength),
        "sparse_nonzeros": int(A_norm.nnz),
        "scale_distance": float(scale)
    }
    with open(str(out_dir / "adj_meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)
    # also copy node_list for downstream mapping
    with open(str(out_dir / "node_list.json"), "w") as f:
        json.dump(node_list, f)
    print("[done] saved adjacency outputs to:", out_dir)
