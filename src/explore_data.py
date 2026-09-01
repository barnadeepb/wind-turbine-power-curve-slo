"""Quick structural check on the downloaded EDP SCADA data — confirms row/column
counts and date ranges match what the plan assumed before any modeling starts.
"""

import pathlib
import pandas as pd

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"

TURBINES = ["T01", "T06", "T07", "T11"]


def main() -> None:
    total_rows = 0
    for t in TURBINES:
        path = RAW_DIR / f"{t}_scada.xlsx"
        df = pd.read_excel(path, sheet_name=0, nrows=5)
        full = pd.read_excel(path, sheet_name=0, usecols=[0])
        n_rows = len(full)
        total_rows += n_rows
        print(f"{t}: {n_rows:,} rows, {df.shape[1]} columns")
        print(f"  columns (first 8): {list(df.columns[:8])}")
        print(f"  date range: {full.iloc[0, 0]} -> {full.iloc[-1, 0]}")

    print(f"\nTotal SCADA rows across 4 turbines: {total_rows:,}")

    fail = pd.read_excel(RAW_DIR / "failure_logbook.xlsx")
    print(f"\nFailure logbook: {len(fail)} rows, columns: {list(fail.columns)}")

    met = pd.read_excel(RAW_DIR / "met_mast_scada_combined.xlsx", nrows=5)
    print(f"\nMet mast/SCADA combined: {met.shape[1]} columns")
    print(f"  columns: {list(met.columns)}")


if __name__ == "__main__":
    main()
