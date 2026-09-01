"""Phase 2a: MLP on GPU. Meant to run concurrently with train_phase2_gp.py
(CPU-bound) — different hardware, no contention.
"""

import json
import pathlib
import time

import numpy as np
import pandas as pd

from mlp_model import train_mlp
from evaluate import regression_metrics, energy_loss_kwh

PROC_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"

FEATURES = [
    "wind_speed_corrected", "Amb_Temp_Avg", "Blds_PitchAngle_Avg",
    "Rtr_RPM_Avg", "Gen_RPM_Avg",
]
TARGET = "Prod_LatestAvg_TotActPwr"


def main() -> None:
    train = pd.read_parquet(PROC_DIR / "train.parquet")
    test = pd.read_parquet(PROC_DIR / "test.parquet")

    # carve a validation slice out of 2016 (last ~15% by time) for early
    # stopping — keeps 2017 untouched as the only number that gets reported
    train = train.sort_values("Timestamp")
    n_val = int(len(train) * 0.15)
    tr, val = train.iloc[:-n_val], train.iloc[-n_val:]
    print(f"train: {len(tr):,} | val: {len(val):,} | test: {len(test):,}")

    # standardize features (helps MLP training; tree models didn't need this)
    feat_mean = tr[FEATURES].mean()
    feat_std = tr[FEATURES].std().replace(0, 1)

    def prep(df):
        return ((df[FEATURES] - feat_mean) / feat_std).values.astype(np.float32)

    X_tr, y_tr = prep(tr), tr[TARGET].values.astype(np.float32)
    X_val, y_val = prep(val), val[TARGET].values.astype(np.float32)
    X_test, y_test = prep(test), test[TARGET].values.astype(np.float32)

    t0 = time.time()
    model, device = train_mlp(X_tr, y_tr, X_val, y_val, epochs=80, patience=10)
    train_s = time.time() - t0
    print(f"device used: {device}")

    import torch
    model.eval()
    X_test_t = torch.tensor(X_test, device=device)
    infer_times = []
    with torch.no_grad():
        for _ in range(10):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred_t = model(X_test_t)
            if device == "cuda":
                torch.cuda.synchronize()
            infer_times.append(time.perf_counter() - t0)
        pred = pred_t.cpu().numpy()
    infer_total = float(np.median(infer_times))
    infer_ms = (infer_total / len(y_test)) * 1000

    result = {
        **regression_metrics(y_test, pred),
        **energy_loss_kwh(y_test, pred),
        "train_seconds": train_s,
        "inference_seconds_total": infer_total,
        "inference_ms_per_sample": infer_ms,
        "device": device,
    }
    print("mlp:", result)

    RESULTS_DIR.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "metrics" / "phase2_mlp.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez(RESULTS_DIR / "metrics" / "phase2_mlp_predictions.npz",
             wind_speed_test=test["wind_speed_corrected"].values, y_test=y_test, mlp=pred)
    print("Saved results/metrics/phase2_mlp.json and predictions")


if __name__ == "__main__":
    main()
