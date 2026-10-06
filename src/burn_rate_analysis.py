"""Burn-rate analysis on the 2017 test set, using the fitted GP's stored
predictions (phase2_gp_predictions.npz, same row order as test.parquet).

Differences from the first submission, all corrections:
  * compliance is the paper's two-sided test, |P - mu| <= k*sigma (the old
    figure code only counted readings *below* the band);
  * windows are real clock time (6 h, 3 d) over the readings actually present,
    trailing (causal), not 36/432 samples with a centred kernel;
  * calibration (empirical +-k*sigma coverage) and a per-turbine / out-of-case
    baseline burn rate are reported, so the case study can be read against
    what "normal" looks like for this model.

Writes results/metrics/burn_rate_analysis.json and burn_rate_case_study.npz.
"""

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"

K = 2.0
THETA = 0.98
SHORT, LONG = "6h", "3D"
MIN_READINGS = 12            # at least 2 h of data inside a window, else undefined
CASE_TURBINE = "T07"
FAILURE_TS = pd.Timestamp("2017-08-20 06:08", tz="UTC")
CASE_START = pd.Timestamp("2017-08-10", tz="UTC")
WARMUP_START = pd.Timestamp("2017-08-07", tz="UTC")   # so the 3-day window is full by CASE_START


def load() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "test.parquet").reset_index(drop=True)
    g = np.load(METRICS / "phase2_gp_predictions.npz")
    assert np.allclose(df["Prod_LatestAvg_TotActPwr"].values, g["y_test"], atol=1e-3), "row order mismatch"
    df["mu"], df["sigma"] = g["gp_mean"], g["gp_std"]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df["outside"] = (df["Prod_LatestAvg_TotActPwr"] - df["mu"]).abs() > K * df["sigma"]
    return df


def trailing_burn(t: pd.DataFrame) -> pd.DataFrame:
    s = t.sort_values("Timestamp").set_index("Timestamp")["outside"].astype(float)
    out = {}
    for name, w in (("short", SHORT), ("long", LONG)):
        frac = s.rolling(w).mean()
        n = s.rolling(w).count()
        out[name] = (frac / (1 - THETA)).where(n >= MIN_READINGS)
    return pd.DataFrame(out)


def main() -> None:
    df = load()
    res = {"k": K, "theta": THETA}
    res["coverage_two_sided"] = float(1 - df["outside"].mean())
    res["coverage_1sigma"] = float(((df["Prod_LatestAvg_TotActPwr"] - df["mu"]).abs() <= df["sigma"]).mean())
    res["overall_burn_rate_if_stationary"] = float(df["outside"].mean() / (1 - THETA))
    z = ((df["Prod_LatestAvg_TotActPwr"] - df["mu"]).abs() / df["sigma"])
    # diagnostic only: the multiplier that WOULD give 95% / 98% two-sided coverage on the
    # 2017 test set. Not used for any reported result (that would be tuning on test).
    res["diagnostic_k_for_95pct_coverage_test"] = float(z.quantile(0.95))
    res["diagnostic_k_for_98pct_coverage_test"] = float(z.quantile(0.98))
    res["by_turbine_outside_pct"] = {t: float(s["outside"].mean() * 100) for t, s in df.groupby("Turbine_ID")}

    t7 = df[df["Turbine_ID"] == CASE_TURBINE].sort_values("Timestamp")
    burn = trailing_burn(t7)
    win = t7[(t7["Timestamp"] >= CASE_START) & (t7["Timestamp"] <= FAILURE_TS)]
    last24 = t7[(t7["Timestamp"] >= FAILURE_TS - pd.Timedelta("24h")) & (t7["Timestamp"] < FAILURE_TS)]
    rest = t7[(t7["Timestamp"] < CASE_START) | (t7["Timestamp"] > FAILURE_TS + pd.Timedelta("2D"))]
    bw = burn.loc[(burn.index >= CASE_START) & (burn.index <= FAILURE_TS)]
    b24 = burn.loc[(burn.index >= FAILURE_TS - pd.Timedelta("24h")) & (burn.index < FAILURE_TS)]
    res["case"] = {
        "turbine": CASE_TURBINE,
        "record_start": str(win["Timestamp"].min()), "record_end": str(win["Timestamp"].max()),
        "n_readings": int(len(win)),
        "outside_pct_window": float(win["outside"].mean() * 100),
        "outside_pct_last24h": float(last24["outside"].mean() * 100),
        "outside_pct_rest_of_2017": float(rest["outside"].mean() * 100),
        "mean_short_burn_window": float(bw["short"].mean()),
        "peak_short_burn_window": float(bw["short"].max()),
        "peak_short_burn_at": str(bw["short"].idxmax()),
        "mean_short_burn_last24h": float(b24["short"].mean()),
        "mean_long_burn_last24h": float(b24["long"].mean()),
        "baseline_burn_rest_of_2017": float(rest["outside"].mean() / (1 - THETA)),
        "days_with_data_gap_over_2h": int((t7.loc[win.index, "Timestamp"].diff() > pd.Timedelta("2h")).sum()),
    }
    # how often does the *rest of the year* exceed the pre-failure level? (alert-threshold sanity)
    r = burn.loc[(burn.index < CASE_START) | (burn.index > FAILURE_TS + pd.Timedelta("2D"))]
    for thr in (10, 15, 20):
        res["case"][f"share_of_rest_short_burn_above_{thr}x"] = float((r["short"] > thr).mean())

    with open(METRICS / "burn_rate_analysis.json", "w") as f:
        json.dump(res, f, indent=2)

    show = t7[(t7["Timestamp"] >= WARMUP_START) & (t7["Timestamp"] <= FAILURE_TS)]
    b = burn.loc[show["Timestamp"]]
    np.savez(
        METRICS / "burn_rate_case_study.npz",
        timestamp=show["Timestamp"].dt.tz_convert(None).values.astype("datetime64[ns]").astype(str),
        actual_kw=show["Prod_LatestAvg_TotActPwr"].values, expected_kw=show["mu"].values,
        std_kw=show["sigma"].values, short_burn=b["short"].values, long_burn=b["long"].values,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
