"""Phase 1: IEC baseline, parametric fit, Random Forest, XGBoost.
All CPU-only, run sequentially — each already uses all available cores
internally, so running them concurrently would only add contention.

Measures training time and per-sample inference latency separately (10
repeated inference passes over the full test set, median reported), and
saves predictions for figure generation.
"""

import json
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from baseline_curves import iec_binned_curve, predict_binned, fit_logistic_curve, predict_logistic
from evaluate import regression_metrics, energy_loss_kwh

PROC_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"

FEATURES = [
    "wind_speed_corrected", "Amb_Temp_Avg", "Blds_PitchAngle_Avg",
    "Rtr_RPM_Avg", "Gen_RPM_Avg",
]
TARGET = "Prod_LatestAvg_TotActPwr"
N_INFER_REPEATS = 10


def time_inference(predict_fn, n_test, repeats=N_INFER_REPEATS):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        pred = predict_fn()
        times.append(time.perf_counter() - t0)
    median_total = float(np.median(times))
    return pred, median_total, (median_total / n_test) * 1000  # ms/sample


def main() -> None:
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(PROC_DIR / "train.parquet")
    test = pd.read_parquet(PROC_DIR / "test.parquet")
    print(f"train: {len(train):,} rows | test: {len(test):,} rows")

    X_train, y_train = train[FEATURES].values, train[TARGET].values
    X_test, y_test = test[FEATURES].values, test[TARGET].values
    ws_train, ws_test = train["wind_speed_corrected"].values, test["wind_speed_corrected"].values
    n_test = len(y_test)

    results = {}
    preds_for_figures = {"wind_speed_test": ws_test, "y_test": y_test}

    # 1. IEC binned curve
    t0 = time.time()
    centers, means, _ = iec_binned_curve(ws_train, y_train)
    train_s = time.time() - t0
    pred, infer_total, infer_ms = time_inference(lambda: predict_binned(ws_test, centers, means), n_test)
    results["iec_binned"] = {
        **regression_metrics(y_test, pred), **energy_loss_kwh(y_test, pred),
        "train_seconds": train_s, "inference_seconds_total": infer_total, "inference_ms_per_sample": infer_ms,
    }
    preds_for_figures["iec_binned"] = pred
    print("iec_binned:", results["iec_binned"])

    # 2. Parametric logistic fit
    t0 = time.time()
    params = fit_logistic_curve(ws_train, y_train)
    train_s = time.time() - t0
    pred, infer_total, infer_ms = time_inference(lambda: predict_logistic(ws_test, params), n_test)
    results["parametric_logistic"] = {
        **regression_metrics(y_test, pred), **energy_loss_kwh(y_test, pred),
        "train_seconds": train_s, "inference_seconds_total": infer_total, "inference_ms_per_sample": infer_ms,
        "params": list(map(float, params)),
    }
    preds_for_figures["parametric_logistic"] = pred
    print("parametric_logistic:", results["parametric_logistic"])

    # 3. Random Forest
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=200, max_depth=16, n_jobs=-1, random_state=0)
    rf.fit(X_train, y_train)
    train_s = time.time() - t0
    pred, infer_total, infer_ms = time_inference(lambda: rf.predict(X_test), n_test)
    results["random_forest"] = {
        **regression_metrics(y_test, pred), **energy_loss_kwh(y_test, pred),
        "train_seconds": train_s, "inference_seconds_total": infer_total, "inference_ms_per_sample": infer_ms,
    }
    preds_for_figures["random_forest"] = pred
    print("random_forest:", results["random_forest"])

    # 4. XGBoost
    t0 = time.time()
    xgb_model = xgb.XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.05, n_jobs=-1, random_state=0)
    xgb_model.fit(X_train, y_train)
    train_s = time.time() - t0
    pred, infer_total, infer_ms = time_inference(lambda: xgb_model.predict(X_test), n_test)
    results["xgboost"] = {
        **regression_metrics(y_test, pred), **energy_loss_kwh(y_test, pred),
        "train_seconds": train_s, "inference_seconds_total": infer_total, "inference_ms_per_sample": infer_ms,
    }
    preds_for_figures["xgboost"] = pred
    print("xgboost:", results["xgboost"])

    with open(RESULTS_DIR / "metrics" / "phase1.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(RESULTS_DIR / "metrics" / "phase1_predictions.npz", **preds_for_figures)
    print("\nSaved results/metrics/phase1.json and phase1_predictions.npz")


if __name__ == "__main__":
    main()
