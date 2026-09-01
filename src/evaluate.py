"""Shared evaluation: regression accuracy + the energy-loss/SLO metrics from the plan."""

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    resid = y_true - y_pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse_kw": rmse, "mae_kw": mae, "r2": r2}


def energy_loss_kwh(y_true: np.ndarray, y_pred: np.ndarray, interval_hours: float = 1 / 6) -> dict:
    """Positive shortfall = actual below expected; interval_hours=1/6 for 10-min readings."""
    shortfall_kw = np.clip(y_pred - y_true, a_min=0, a_max=None)
    total_kwh = float(shortfall_kw.sum() * interval_hours)
    expected_kwh = float(y_pred.sum() * interval_hours)
    pct_of_expected = 100 * total_kwh / expected_kwh if expected_kwh > 0 else float("nan")
    return {"shortfall_kwh": total_kwh, "expected_kwh": expected_kwh, "shortfall_pct": pct_of_expected}
