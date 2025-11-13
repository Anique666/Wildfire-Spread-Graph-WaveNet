
# Project run-order (PowerShell one-liners)

This file contains the exact, minimal PowerShell one-liners to run the full data preparation, feature aggregation, edge dataset creation, baseline training, and spatial visualization used in this project.

Run each command from the project root (the folder that contains the scripts and the `data/` directory). PowerShell note: put each command on one line (don't use `\`).

## 1) Preprocess raw CSVs → `preproc_output`

Creates cleaned, reprojected grid/topo/weather/fire files that later steps consume.

```powershell
python data_pre.py --fire ./data/fire_data.csv --weather ./data/output_final_temp_celsius_fixed.csv --topo ./data/topo_data_cleaned.csv --out_dir ./preproc_output --time_res_minutes 60
```

## 2) Aggregate features into cell × time bins → `agg_output`

Turns raw preproc outputs into `cells_static.parquet`, `weather_cell_time.parquet`, `fire_cell_time.parquet`, and a merged sparse `panel_cell_time.parquet`.

```powershell
python feature_agg.py --preproc_dir ./preproc_output --out_dir ./agg_output --time_res_minutes 60
```

## 3) Post-process aggregates (nearest-station weather mapping, restrict fire cells, kNN topo impute)

Maps weather to nearest real stations, restricts fire rows to panel cells and imputes missing static topo with kNN (k=8).

```powershell
python postproc_agg.py --agg_dir ./agg_output --preproc_dir ./preproc_output --weather_csv ./data/output_final_temp_celsius_fixed.csv --time_res_minutes 60 --k 8
```

## 4) Build directed edge spread dataset (24-hour horizon, neighbor+trend features)

Produces `edge_spread_examples_full.parquet` (all examples), `edge_spread_examples.parquet` (balanced sample), and `edge_feature_list.json` (exact feature order).

```powershell
python edge_dataset.py --panel ./agg_output/panel_cell_time.parquet --grid ./preproc_output/grid.gpkg --static ./agg_output/cells_static.parquet --out ./edge_spread_examples.parquet --k 8 --horizon 24 --lags 6 --neg_ratio 8
```

## 5) Train XGBoost baseline (class weighting; optional focal loss flag available)

Trains model (uses `scale_pos_weight` automatically), saves `xgb_edge_baseline.joblib` and `xgb_feature_list.json`.

```powershell
python baseline.py --edge_data ./edge_spread_examples.parquet --model_out ./xgb_edge_baseline.joblib
```

If you want to train with focal loss instead of class-weighting, add `--use_focal` plus `--alpha` and `--gamma` (example below).

## 6) Visualize spatial predictions vs actual spread for a chosen time (24h horizon)

Creates a side-by-side map: actual new burns in (t, t+24h] vs predicted per-cell spread probability (and arrows for high-prob edges).

```powershell
python baseline_viz_spatial.py --grid ./preproc_output/grid.gpkg --panel ./agg_output/panel_cell_time.parquet --edge ./edge_spread_examples_full.parquet --model ./xgb_edge_baseline.joblib --time_epoch 438298 --horizon 24 --prob_thresh 0.4
```

---

## Quick helpers

To see positive/negative counts after step 4 run:

```powershell
python - <<'PY'
import pandas as pd
df = pd.read_parquet("edge_spread_examples_full.parquet")
print("total:", len(df), "positives:", int(df.label.sum()), "negatives:", len(df)-int(df.label.sum()))
PY
```

To train with focal loss (if `scale_pos_weight` is insufficient) add `--use_focal` in step 5, for example:

```powershell
python baseline.py --edge_data ./edge_spread_examples.parquet --model_out ./xgb_edge_baseline.joblib --use_focal --alpha 0.25 --gamma 2.0
```

## Notes

- File paths in the one-liners assume you run them from the project root.
- `--time_res_minutes` should be consistent across preprocessing, aggregation, and post-processing steps.
- The `edge_dataset.py` command creates both the full and sampled balanced parquet outputs; use the `_full` file for diagnostics and counts.

If you want, I can also commit this updated `readme.md` into version control or add a short section describing the output files and locations in more detail.
