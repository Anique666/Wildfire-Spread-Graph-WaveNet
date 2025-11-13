#!/usr/bin/env python3
"""
train_xgboost_baseline.py

Loads balanced edge dataset and feature list, trains XGBoost and saves both model and feature-list JSON.
python baseline.py --edge_data ./edge_spread_examples.parquet --model_out ./xgb_edge_baseline.joblib
"""
import argparse, joblib, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, classification_report
import xgboost as xgb

def focal_obj(alpha=0.25, gamma=2.0):
    # returns a custom objective for XGBoost (binary focal loss)
    def _focal(preds, dtrain):
        labels = dtrain.get_label()
        # preds are raw scores (logits)
        p = 1.0 / (1.0 + np.exp(-preds))
        grad = (p - labels) * ((alpha * labels + (1-alpha)*(1-labels)) * ((1-p)**gamma))
        # approximate hess: p*(1-p) scaled
        hess = p * (1.0 - p) * ((alpha * labels + (1-alpha)*(1-labels)) * ((1-p)**gamma))
        return grad, hess
    return _focal

def train(edge_parquet, model_out="xgb_edge_baseline.joblib", feature_json=None, use_focal=False, alpha=0.25, gamma=2.0):
    t0 = time.time()
    df = pd.read_parquet(edge_parquet)
    # load feature list: prefer sibling JSON
    if feature_json is None:
        feature_json = Path(edge_parquet).with_name("edge_feature_list.json")
    feature_json = Path(feature_json)
    if not feature_json.exists():
        # fallback: infer numeric columns
        feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ("label","time_epoch","sx","sy","tx","ty")]
        print("[warn] feature json missing; inferring numeric features (not ideal).")
    else:
        feat_cols = json.loads(feature_json.read_text())
    print("[info] using feature columns (first 20):", feat_cols[:20], " total:", len(feat_cols))
    X = df[feat_cols].fillna(0.0).values
    y = df["label"].values
    # train/test split (time-respecting if possible)
    if "time_epoch" in df.columns:
        times = np.sort(df["time_epoch"].unique())
        if len(times) > 10:
            t1 = int(np.percentile(times, 70)); t2 = int(np.percentile(times, 85))
            train_idx = df["time_epoch"] <= t1
            val_idx = (df["time_epoch"] > t1) & (df["time_epoch"] <= t2)
            test_idx = df["time_epoch"] > t2
            if train_idx.sum() < 50:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            else:
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                X_test, y_test = X[test_idx], y[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # compute scale_pos_weight
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    scale_pos = max(1.0, (neg / max(1, pos)))
    print(f"[info] train pos={pos}, neg={neg}, scale_pos_weight={scale_pos:.3f}")
    # model init
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=6,
        scale_pos_weight=scale_pos
    )
    # fit with optional focal loss
    try:
        if use_focal:
            print("[info] Training with focal loss (custom objective).")
            obj = focal_obj(alpha=alpha, gamma=gamma)
            # xgboost sklearn wrapper doesn't accept custom objective directly in some versions;
            # fall back to training via xgb.train if needed
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_cols)
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feat_cols)
            params = {
                "eta": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "seed": 42
            }
            bst = xgb.train(params, dtrain, num_boost_round=500, obj=obj, evals=[(dtest,"test")], early_stopping_rounds=25)
            # wrap into sklearn-compatible object by loading booster
            model._Booster = bst
            # note: model.get_booster() might now work, but some sklearn wrappers expect more; saving will work with joblib.dump(bst)
        else:
            print("[info] Training with XGBClassifier (scale_pos_weight used).")
            try:
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=25, verbose=True)
            except TypeError as e:
                # older xgboost version doesn't accept early_stopping_rounds in sklearn API
                print("[warn] early_stopping_rounds not supported by this xgboost; falling back to fit without it.")
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    except Exception as e:
        print("[error] training failed:", e)
        raise

    # save model + feature list JSON
    joblib.dump(model, model_out)
    out_feat_json = Path(model_out).with_name("xgb_feature_list.json")
    with open(out_feat_json, "w") as f:
        json.dump(feat_cols, f)
    # evaluate on test set
    if 'X_test' in locals():
        try:
            y_proba = model.predict_proba(X_test)[:,1]
        except Exception:
            # if model wrapped as booster from xgb.train
            bst = model.get_booster()
            y_proba = bst.predict(xgb.DMatrix(X_test))
    else:
        y_proba = model.predict_proba(X_train)[:,1]
    ap = average_precision_score(y_test, y_proba)
    y_pred = (y_proba >= 0.5).astype(int)
    print("[RESULT] AP:", ap)
    print(classification_report(y_test, y_pred, digits=3))
    print("[saved] model:", model_out, "features:", out_feat_json)
    print("[time] seconds:", time.time()-t0)
    return model, out_feat_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--edge_data", required=True)
    p.add_argument("--model_out", default="./xgb_edge_baseline.joblib")
    p.add_argument("--use_focal", action="store_true", help="Use focal loss (custom objective) instead of plain xgb classifier")
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=2.0)
    args = p.parse_args()
    train(args.edge_data, model_out=args.model_out, use_focal=args.use_focal, alpha=args.alpha, gamma=args.gamma)