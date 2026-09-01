"""Export train/test data to CSV for Vertex AI AutoML, with an explicit
predefined split column so AutoML uses our time-based 2016/2017 split
instead of its own random split — the whole point of the split choice was
avoiding leakage across autocorrelated adjacent readings, and a random
split would reintroduce exactly that.
"""

import pathlib
import pandas as pd

PROC_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"

FEATURES = [
    "wind_speed_corrected", "Amb_Temp_Avg", "Blds_PitchAngle_Avg",
    "Rtr_RPM_Avg", "Gen_RPM_Avg",
]
TARGET = "Prod_LatestAvg_TotActPwr"
COLS = FEATURES + [TARGET]


def main() -> None:
    train = pd.read_parquet(PROC_DIR / "train.parquet")
    test = pd.read_parquet(PROC_DIR / "test.parquet")

    train = train.sort_values("Timestamp")
    n_val = int(len(train) * 0.15)
    tr, val = train.iloc[:-n_val], train.iloc[-n_val:]

    tr = tr[COLS].copy(); tr["ml_use"] = "TRAIN"
    val = val[COLS].copy(); val["ml_use"] = "VALIDATE"
    te = test[COLS].copy(); te["ml_use"] = "TEST"

    combined = pd.concat([tr, val, te], ignore_index=True)
    out_path = PROC_DIR / "automl_combined.csv"
    combined.to_csv(out_path, index=False)
    print(f"wrote {len(combined):,} rows ({len(tr):,} train / {len(val):,} validate / {len(te):,} test) to {out_path}")


if __name__ == "__main__":
    main()
