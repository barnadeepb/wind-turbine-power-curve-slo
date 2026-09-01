"""Phase 1: IEC baseline, parametric fit, Random Forest, XGBoost.
All CPU-only, run sequentially — each already uses all available cores
internally, so running them concurrently would only add contention.
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


def main() -> None:
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(PROC_DIR / "train.parquet")
    test = pd.read_parquet(PROC_DIR / "test.parquet")
    print(f"train: {len(train):,} rows | test: {len(test):,} rows")

    X_train, y_train = train[FEATURES].values, train[TARGET].values
    X_test, y_test = test[FEATURES].values, test[TARGET].values
    ws_train, ws_test = train["wind_speed_corrected"].values, test["wind_speed_corrected"].values

    results = {}

    # 1. IEC binned curve
    t0 = time.time()
    centers, means, _ = iec_binned_curve(ws_train, y_train)
    pred = predict_binned(ws_test, centers, means)
    results["iec_binned"] = {
        **regression_metrics(y_test, pred),
        **energy_loss_kwh(y_test, pred),
        "train_seconds": time.time() - t0,
    }
    print("iec_binned:", results["iec_binned"])

    # 2. Parametric logistic fit
    t0 = time.time()
    params = fit_logistic_curve(ws_train, y_train)
    pred = predict_logistic(ws_test, params)
    results["parametric_logistic"] = {
        **regression_metrics(y_test, pred),
        **energy_loss_kwh(y_test, pred),
        "train_seconds": time.time() - t0,
        "params": list(map(float, params)),
    }
    print("parametric_logistic:", results["parametric_logistic"])

    # 3. Random Forest
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=200, max_depth=16, n_jobs=-1, random_state=0)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    results["random_forest"] = {
        **regression_metrics(y_test, pred),
        **energy_loss_kwh(y_test, pred),
        "train_seconds": time.time() - t0,
    }
    print("random_forest:", results["random_forest"])

    # 4. XGBoost
    t0 = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=400, max_depth=8, learning_rate=0.05,
        n_jobs=-1, random_state=0,
    )
    xgb_model.fit(X_train, y_train)
    pred = xgb_model.predict(X_test)
    results["xgboost"] = {
        **regression_metrics(y_test, pred),
        **energy_loss_kwh(y_test, pred),
        "train_seconds": time.time() - t0,
    }
    print("xgboost:", results["xgboost"])

    with open(RESULTS_DIR / "metrics" / "phase1.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/metrics/phase1.json")


if __name__ == "__main__":
    main()
