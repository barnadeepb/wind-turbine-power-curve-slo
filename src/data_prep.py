"""Build the combined, feature-ready dataset from raw SCADA + met mast files.

Output: data/processed/{train,test}.parquet, split by calendar year
(2016 train / 2017 test) to avoid leakage between autocorrelated adjacent
10-minute readings and to keep full seasonal coverage on both sides.
"""

import pathlib
import numpy as np
import pandas as pd

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"

TURBINES = ["T01", "T06", "T07", "T11"]

SCADA_COLS = [
    "Turbine_ID", "Timestamp",
    "Amb_WindSpeed_Avg", "Amb_Temp_Avg", "Amb_WindDir_Abs_Avg",
    "Blds_PitchAngle_Avg", "Rtr_RPM_Avg", "Gen_RPM_Avg",
    "Prod_LatestAvg_TotActPwr", "Grd_Prod_PsblePwr_Avg",
    "Nac_Direction_Avg",
]

# R_specific for dry air, J/(kg*K)
R_SPECIFIC = 287.05


def air_density(temp_c: pd.Series, pressure_pa: pd.Series) -> pd.Series:
    """Ideal-gas air density, per the correction IEC 61400-12-1 requires."""
    temp_k = temp_c + 273.15
    return pressure_pa / (R_SPECIFIC * temp_k)


def load_scada() -> pd.DataFrame:
    frames = []
    for t in TURBINES:
        path = RAW_DIR / f"{t}_scada.xlsx"
        df = pd.read_excel(path, usecols=SCADA_COLS)
        frames.append(df)
        print(f"  loaded {t}: {len(df):,} rows")
    return pd.concat(frames, ignore_index=True)


def load_met_mast() -> pd.DataFrame:
    df = pd.read_excel(
        RAW_DIR / "met_mast_scada_combined.xlsx",
        usecols=["Timestamp", "Avg_AmbientTemp", "Avg_Pressure"],
    )
    return df.rename(columns={"Avg_AmbientTemp": "mast_temp_c", "Avg_Pressure": "mast_pressure_pa"})


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SCADA...")
    scada = load_scada()
    scada["Timestamp"] = pd.to_datetime(scada["Timestamp"], utc=True)

    print("Loading met mast...")
    mast = load_met_mast()
    mast["Timestamp"] = pd.to_datetime(mast["Timestamp"], utc=True)

    print("Merging on timestamp (nearest, tolerance 5 min)...")
    scada = scada.sort_values("Timestamp")
    mast = mast.sort_values("Timestamp")
    df = pd.merge_asof(
        scada, mast, on="Timestamp",
        direction="nearest", tolerance=pd.Timedelta("5min"),
    )

    before = len(df)
    df = df.dropna(subset=["mast_pressure_pa", "mast_temp_c", "Amb_WindSpeed_Avg", "Prod_LatestAvg_TotActPwr"])
    print(f"Dropped {before - len(df):,} rows with missing pressure/temp/wind/power ({len(df):,} remain)")

    # source column is watts; keep everything downstream in kW
    df["Prod_LatestAvg_TotActPwr"] = df["Prod_LatestAvg_TotActPwr"] / 1000.0

    df["air_density"] = air_density(df["mast_temp_c"], df["mast_pressure_pa"])
    # IEC 61400-12-1 normalization to reference density (1.225 kg/m^3)
    df["wind_speed_corrected"] = df["Amb_WindSpeed_Avg"] * (df["air_density"] / 1.225) ** (1 / 3)

    # NOTE: Grd_Prod_PsblePwr_Avg does not track real-time achievable power —
    # it stays near rated capacity regardless of wind conditions, so it can't
    # be used to separate curtailment from genuine underperformance (verified
    # against training data: 100% of rows would "flag" under any reasonable
    # threshold, which just reflects normal wind capacity factor, not
    # curtailment). Kept in the data for reference; not used to filter.
    # See README limitations — curtailment is not separated out in this pass.

    df["year"] = df["Timestamp"].dt.year
    train = df[df["year"] == 2016].copy()
    test = df[df["year"] == 2017].copy()

    print(f"\nTrain (2016): {len(train):,} rows")
    print(f"Test  (2017): {len(test):,} rows")

    train.to_parquet(PROC_DIR / "train.parquet", index=False)
    test.to_parquet(PROC_DIR / "test.parquet", index=False)
    print("\nSaved to data/processed/")


if __name__ == "__main__":
    main()
