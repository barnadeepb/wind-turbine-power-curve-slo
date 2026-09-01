"""IEC 61400-12-1 binned power curve and a parametric logistic fit.

Both operate on plain wind_speed/power arrays so they don't depend on the
raw SCADA column names — src/data_prep.py is responsible for producing
those arrays (with air-density correction applied) before these are called.
"""

import numpy as np
from scipy.optimize import curve_fit


def iec_binned_curve(wind_speed: np.ndarray, power: np.ndarray, bin_width: float = 0.5):
    """Bin-average power curve per IEC 61400-12-1. Returns (bin_centers, mean_power, bin_edges)."""
    bins = np.arange(0, wind_speed.max() + bin_width, bin_width)
    bin_idx = np.digitize(wind_speed, bins)
    centers, means = [], []
    for i in range(1, len(bins)):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        centers.append((bins[i - 1] + bins[i]) / 2)
        means.append(power[mask].mean())
    return np.array(centers), np.array(means), bins


def predict_binned(wind_speed: np.ndarray, bin_centers: np.ndarray, bin_means: np.ndarray) -> np.ndarray:
    """Nearest-bin lookup for arbitrary wind speeds."""
    idx = np.clip(np.searchsorted(bin_centers, wind_speed), 0, len(bin_centers) - 1)
    return bin_means[idx]


def _logistic(v, p_rated, v0, k):
    return p_rated / (1 + np.exp(-k * (v - v0)))


def fit_logistic_curve(wind_speed: np.ndarray, power: np.ndarray):
    """Parametric logistic power curve, fit by least squares on raw (not binned) points."""
    p_rated_guess = np.percentile(power, 99)
    v0_guess = np.median(wind_speed)
    params, _ = curve_fit(
        _logistic, wind_speed, power,
        p0=[p_rated_guess, v0_guess, 1.0],
        maxfev=10000,
    )
    return params  # (p_rated, v0, k)


def predict_logistic(wind_speed: np.ndarray, params) -> np.ndarray:
    return _logistic(wind_speed, *params)
