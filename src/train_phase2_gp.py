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
    print(f"actual GP training set size: {fitted['n_train']:,}")

    mean_pred, std_pred = predict(fitted, ws_test)

    result = {
        **regression_metrics(y_test, mean_pred),
        **energy_loss_kwh(y_test, mean_pred),
        "train_seconds": time.time() - t0,
        "n_gp_points": fitted["n_train"],
        "mean_predictive_std_kw": float(std_pred.mean()),
    }
    print("heteroscedastic_gp:", result)

    RESULTS_DIR.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "metrics" / "phase2_gp.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved results/metrics/phase2_gp.json")


if __name__ == "__main__":
    main()
