"""Phase 2b: heteroscedastic GP on CPU. Meant to run concurrently with
train_phase2_mlp.py (GPU-bound) — different hardware, no contention.

This is the one to watch for stalls (hyperparameter optimization can run
long or fail to converge) — progress prints every 25 GP iterations across
all three fitting stages, so silence for an extended stretch is the signal
to check on it, not just "still running."
"""

import json
import pathlib
import time

import numpy as np
import pandas as pd

from gp_model import fit_heteroscedastic_gp, predict
from evaluate import regression_metrics, energy_loss_kwh

PROC_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"

TARGET = "Prod_LatestAvg_TotActPwr"
N_SUBSAMPLE = 15000


def main() -> None:
    train = pd.read_parquet(PROC_DIR / "train.parquet")
    test = pd.read_parquet(PROC_DIR / "test.parquet")
    print(f"train pool: {len(train):,} | test: {len(test):,} | GP subsample target: {N_SUBSAMPLE:,}")

    ws_train = train["wind_speed_corrected"].values
    y_train = train[TARGET].values
    ws_test = test["wind_speed_corrected"].values
    y_test = test[TARGET].values

    t0 = time.time()
    fitted = fit_heteroscedastic_gp(ws_train, y_train, n_subsample=N_SUBSAMPLE)
    train_s = time.time() - t0
    print(f"actual GP training set size: {fitted['n_train']:,}")

    infer_times = []
    for _ in range(3):  # GP inference is slow enough that 3 reps is plenty for a stable median
        t0 = time.time()
        mean_pred, std_pred = predict(fitted, ws_test)
        infer_times.append(time.time() - t0)
    infer_total = float(np.median(infer_times))
    infer_ms = (infer_total / len(y_test)) * 1000

    result = {
        **regression_metrics(y_test, mean_pred),
        **energy_loss_kwh(y_test, mean_pred),
        "train_seconds": train_s,
        "inference_seconds_total": infer_total,
        "inference_ms_per_sample": infer_ms,
        "n_gp_points": fitted["n_train"],
        "mean_predictive_std_kw": float(std_pred.mean()),
    }
    print("heteroscedastic_gp:", result)

    RESULTS_DIR.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "metrics" / "phase2_gp.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez(
        RESULTS_DIR / "metrics" / "phase2_gp_predictions.npz",
        wind_speed_test=ws_test, y_test=y_test, gp_mean=mean_pred, gp_std=std_pred,
    )
    print("Saved results/metrics/phase2_gp.json and predictions")

    # --- burn-rate case study: T07, the bearing-damage -> generator-damage
    # cascade on 2017-08-20/21, using the same fitted GP (predictive std as
    # the SLO tolerance band) on a real contiguous time window ---
    print("\nBuilding burn-rate case study for T07 2017-08-10..2017-08-25 ...")
    window = test[
        (test["Turbine_ID"] == "T07")
        & (test["Timestamp"] >= "2017-08-10")
        & (test["Timestamp"] < "2017-08-26")
    ].sort_values("Timestamp")
    print(f"window rows: {len(window):,}")
    if len(window) > 0:
        w_mean, w_std = predict(fitted, window["wind_speed_corrected"].values)
        np.savez(
            RESULTS_DIR / "metrics" / "burn_rate_case_study.npz",
            timestamp=window["Timestamp"].values.astype("datetime64[ns]").astype(str),
            actual_kw=window[TARGET].values,
            expected_kw=w_mean,
            std_kw=w_std,
        )
        print("Saved results/metrics/burn_rate_case_study.npz")
    else:
        print("WARNING: no rows found for T07 case-study window")


if __name__ == "__main__":
    main()
