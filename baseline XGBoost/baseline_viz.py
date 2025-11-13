#!/usr/bin/env python3
"""
visualize_xgb_results.py

Visualizes XGBoost model performance and interpretability:
 - Feature importance (Gain, Cover, Weight)
 - Precision-Recall curve
 - Confusion matrix
 - Calibration reliability plot

Usage:
python baseline_viz.py --edge_data ./edge_spread_examples.parquet --model ./xgb_edge_baseline.joblib
"""

import argparse
import json
import joblib
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

# ---------------------- MAIN ---------------------- #

def _load_feature_list_for_model(model_path, explicit_feat_json=None, edge_data_path=None):
    # Priority: explicit_feat_json -> sibling xgb_feature_list.json -> infer from edge_data
    if explicit_feat_json:
        p = Path(explicit_feat_json)
        if p.exists():
            return json.loads(p.read_text())
    m = Path(model_path)
    sibling = m.with_name("xgb_feature_list.json")
    if sibling.exists():
        return json.loads(sibling.read_text())
    # fallback: infer from edge_data if provided
    if edge_data_path:
        df = pd.read_parquet(edge_data_path)
        feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ("label", "time_epoch", "sx", "sy", "tx", "ty")]
        return feat_cols
    return None


def _safe_predict_proba(model, X, feat_cols=None):
    # Try sklearn predict_proba, then xgboost booster predict
    try:
        probs = model.predict_proba(X)[:, 1]
        return probs
    except Exception:
        pass
    # try model.get_booster() or model._Booster
    booster = None
    try:
        booster = model.get_booster()
    except Exception:
        booster = getattr(model, "_Booster", None)
    if booster is not None:
        try:
            dmat = xgb.DMatrix(X, feature_names=feat_cols)
            probs = booster.predict(dmat)
            return probs
        except Exception:
            # last resort: predict without feature names
            dmat = xgb.DMatrix(X)
            probs = booster.predict(dmat)
            return probs
    raise RuntimeError("Unable to get prediction probabilities from model")


def _get_feature_importance(model, feat_cols=None, topn=20):
    # Return ordered dict {feat:score}
    # Prefer booster.get_score
    booster = None
    try:
        booster = model.get_booster()
    except Exception:
        booster = getattr(model, "_Booster", None)
    if booster is not None:
        imps = booster.get_score(importance_type="gain")
        # imps keys may be f0,f1... or real names
        if not imps:
            return {}
        # if keys are f0.. map to feat_cols
        keys = list(imps.keys())
        if feat_cols and all(k.startswith("f") for k in keys) and len(keys) == len(feat_cols):
            mapped = {feat_cols[int(k[1:])]: v for k, v in imps.items()}
        else:
            mapped = imps
        # sort and take topn
        mapped = dict(sorted(mapped.items(), key=lambda x: x[1], reverse=True)[:topn])
        return mapped
    # fallback: sklearn feature_importances_
    if hasattr(model, "feature_importances_") and feat_cols is not None:
        arr = np.asarray(model.feature_importances_)
        if len(arr) == len(feat_cols):
            pairs = list(zip(feat_cols, arr))
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:topn]
            return dict(pairs)
    return {}


def visualize(edge_data, model_path, feat_json=None, topn=20, save_fig_dir=None):
    model_path = Path(model_path)
    print(f"[INFO] Loading model: {model_path}")
    model = joblib.load(str(model_path))
    print(f"[INFO] Loading dataset: {edge_data}")
    df = pd.read_parquet(edge_data)

    # Determine feature columns
    feat_cols = _load_feature_list_for_model(model_path, explicit_feat_json=feat_json, edge_data_path=edge_data)
    if feat_cols is None:
        # final fallback: numeric columns except label
        feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "label"]
        print("[warn] No feature list found; inferring numeric features.")

    # ensure features exist in df
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"[warn] Some feature columns missing from edge data and will be dropped: {missing[:10]}")
        feat_cols = [c for c in feat_cols if c in df.columns]

    X = df[feat_cols].fillna(0.0).values
    y = df["label"].values

    # get predictions safely
    y_pred_proba = _safe_predict_proba(model, X, feat_cols=feat_cols)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # --- 1. Feature importance ---
    print("[INFO] Computing feature importance...")
    importance_gain = _get_feature_importance(model, feat_cols=feat_cols, topn=topn)
    # if mapped keys are f0.., they were converted above where possible

    if importance_gain:
        plt.figure(figsize=(8, 6))
        plt.barh(list(importance_gain.keys()), list(importance_gain.values()))
        plt.xlabel("Importance (Gain)")
        plt.ylabel("Feature")
        plt.title(f"Top {topn} Feature Importances (Gain)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if save_fig_dir:
            Path(save_fig_dir).mkdir(parents=True, exist_ok=True)
            out = Path(save_fig_dir) / "feature_importance_gain.png"
            plt.savefig(out, dpi=200)
            print(f"[saved] {out}")
            plt.close()
        else:
            plt.show()
    else:
        print("[warn] No feature importance available for this model.")

    # --- 2. Precision–Recall Curve ---
    print("[INFO] Plotting precision-recall curve...")
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    ap = average_precision_score(y, y_pred_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="blue", lw=2)
    plt.fill_between(recall, precision, alpha=0.2, color="blue")
    plt.title(f"Precision–Recall Curve (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    if save_fig_dir:
        out = Path(save_fig_dir) / "precision_recall.png"
        plt.savefig(out, dpi=200)
        print(f"[saved] {out}")
        plt.close()
    else:
        plt.show()

    # --- 3. Confusion Matrix ---
    print("[INFO] Plotting confusion matrix...")
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Spread", "Spread"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (threshold=0.5)")
    if save_fig_dir:
        out = Path(save_fig_dir) / "confusion_matrix.png"
        plt.savefig(out, dpi=200)
        print(f"[saved] {out}")
        plt.close()
    else:
        plt.show()

    # --- 4. Calibration Curve ---
    print("[INFO] Plotting calibration curve...")
    prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="XGBoost")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_fig_dir:
        out = Path(save_fig_dir) / "calibration_curve.png"
        plt.savefig(out, dpi=200)
        print(f"[saved] {out}")
        plt.close()
    else:
        plt.show()

    # --- 5. Text summary ---
    print("\n[INFO] Classification Report:")
    print(classification_report(y, y_pred, digits=3))
    print(f"Average Precision (PR-AUC): {ap:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_data", required=True, help="Path to edge_spread_examples.parquet")
    ap.add_argument("--model", required=True, help="Path to trained xgb_edge_baseline.joblib")
    ap.add_argument("--topn", type=int, default=20, help="Top-N features for importance plot")
    args = ap.parse_args()

    visualize(args.edge_data, args.model, topn=args.topn)
