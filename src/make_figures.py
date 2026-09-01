"""Generate SVG figures from real model outputs — no illustrative/fabricated
data. Requires phase1_predictions.npz, phase2_mlp_predictions.npz,
phase2_gp_predictions.npz, burn_rate_case_study.npz, and the three metrics
JSON files (phase1/phase2_mlp/phase2_gp/phase3_automl) to already exist.
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = RESULTS_DIR / "metrics"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
})

MODEL_ORDER = ["iec_binned", "parametric_logistic", "random_forest", "xgboost", "mlp", "heteroscedastic_gp", "automl"]
MODEL_LABELS = {
    "iec_binned": "IEC\nbinned", "parametric_logistic": "Parametric\nlogistic",
    "random_forest": "Random\nForest", "xgboost": "XGBoost", "mlp": "MLP",
    "heteroscedastic_gp": "Heteroscedastic\nGP", "automl": "AutoML\n(Vertex AI)",
}


def load_all_metrics():
    p1 = json.load(open(METRICS_DIR / "phase1.json"))
    mlp = json.load(open(METRICS_DIR / "phase2_mlp.json"))
    gp = json.load(open(METRICS_DIR / "phase2_gp.json"))
    automl = json.load(open(METRICS_DIR / "phase3_automl.json"))
    combined = dict(p1)
    combined["mlp"] = mlp
    combined["heteroscedastic_gp"] = gp
    combined["automl"] = {
        "rmse_kw": automl["vertex_metrics"]["rootMeanSquaredError"],
        "r2": automl["vertex_metrics"]["rSquared"],
        "mae_kw": automl["vertex_metrics"]["meanAbsoluteError"],
        "train_seconds": automl["train_seconds"],
        "inference_ms_per_sample": None,
    }
    return combined


def fig_accuracy_comparison(metrics):
    # horizontal bars, stacked vertically — legible at single-column width,
    # unlike side-by-side vertical bars where 7 model names collide
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.6))
    labels = [MODEL_LABELS[m].replace("\n", " ") for m in MODEL_ORDER]
    y = np.arange(len(labels))
    rmse = [metrics[m]["rmse_kw"] for m in MODEL_ORDER]
    r2 = [metrics[m]["r2"] for m in MODEL_ORDER]
    colors = ["#8a8a8a"] * 5 + ["#2E7D4F", "#B23A2E"]

    ax = axes[0]
    bars = ax.barh(y, rmse, color=colors, height=0.65)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (kW)", fontsize=7)
    ax.set_title("(a) Test-set RMSE", fontsize=8)
    ax.tick_params(axis="x", labelsize=6)
    for b, v in zip(bars, rmse):
        ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=6)

    ax = axes[1]
    bars = ax.barh(y, r2, color=colors, height=0.65)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0.65, 1.05)
    ax.set_xlabel("R²", fontsize=7)
    ax.set_title("(b) Test-set R²", fontsize=8)
    ax.tick_params(axis="x", labelsize=6)
    for b, v in zip(bars, r2):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=6)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_accuracy_comparison.svg"); fig.savefig(FIG_DIR / "fig_accuracy_comparison.png", dpi=220)
    plt.close(fig)
    print("wrote fig_accuracy_comparison.svg")


def fig_cost_comparison(metrics):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    order = [m for m in MODEL_ORDER if m != "automl"]  # automl inference latency not measured
    labels = [MODEL_LABELS[m] for m in order]
    train_s = [metrics[m]["train_seconds"] for m in order]
    colors = ["#8a8a8a"] * 5 + ["#2E7D4F"]
    bars = ax.bar(labels, train_s, color=colors, width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Training time (s, log scale)")
    ax.tick_params(axis="x", labelsize=6.5)
    for b, v in zip(bars, train_s):
        label = f"{v:.2f}s" if v < 60 else f"{v/60:.1f}min"
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, label, ha="center", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cost_comparison.svg"); fig.savefig(FIG_DIR / "fig_cost_comparison.png", dpi=220)
    plt.close(fig)
    print("wrote fig_cost_comparison.svg")


def fig_power_curve():
    p1 = np.load(METRICS_DIR / "phase1_predictions.npz")
    gp = np.load(METRICS_DIR / "phase2_gp_predictions.npz")
    ws, y = p1["wind_speed_test"], p1["y_test"]

    # thin the scatter for a legible SVG (every 8th point), sort GP curve by wind speed
    idx_scatter = np.arange(0, len(ws), 8)
    order = np.argsort(gp["wind_speed_test"])
    ws_sorted = gp["wind_speed_test"][order]
    mean_sorted = gp["gp_mean"][order]
    std_sorted = gp["gp_std"][order]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.scatter(ws[idx_scatter], y[idx_scatter], s=2, alpha=0.15, color="#555555", label="Test readings")
    ax.plot(ws_sorted, mean_sorted, color="#B23A2E", lw=1.3, label="GP expected power")
    ax.fill_between(
        ws_sorted, mean_sorted - 2 * std_sorted, mean_sorted + 2 * std_sorted,
        color="#B23A2E", alpha=0.2, label="±2σ SLO tolerance band",
    )
    ax.set_xlabel("Corrected wind speed (m/s)")
    ax.set_ylabel("Active power (kW)")
    ax.legend(fontsize=6, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_power_curve_slo.svg"); fig.savefig(FIG_DIR / "fig_power_curve_slo.png", dpi=220)
    plt.close(fig)
    print("wrote fig_power_curve_slo.svg")


def fig_burn_rate_case_study():
    d = np.load(METRICS_DIR / "burn_rate_case_study.npz", allow_pickle=True)
    ts = d["timestamp"].astype("datetime64[ns]")
    actual, expected, std = d["actual_kw"], d["expected_kw"], d["std_kw"]

    k = 2.0
    lower = expected - k * std
    compliant = actual >= lower
    theta = 0.98
    non_compliant = (~compliant).astype(float)
    # rolling 6-hour (36-sample) short window, 3-day (432-sample) long window burn rate
    def rolling_burn_rate(x, window, theta):
        kernel = np.ones(window) / window
        rate = np.convolve(x, kernel, mode="same")
        return rate / (1 - theta)
    short_burn = rolling_burn_rate(non_compliant, 36, theta)
    long_burn = rolling_burn_rate(non_compliant, 432, theta)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    bearing_ts = np.datetime64("2017-08-20T06:08")
    record_end = ts.max()  # data stops here — turbine offline after the logged bearing failure

    ax = axes[0]
    ax.plot(ts, actual, color="#2C5F8A", lw=0.8, label="Actual power")
    ax.plot(ts, expected, color="#B23A2E", lw=0.8, label="GP expected power")
    ax.fill_between(ts, expected - k * std, expected + k * std, color="#B23A2E", alpha=0.15, label="±2σ SLO band")
    ax.axvline(bearing_ts, color="black", lw=0.8, ls="--")
    ax.axvspan(record_end, bearing_ts + np.timedelta64(1, "D"), color="#dddddd", alpha=0.5)
    ax.set_ylabel("Power (kW)")
    ax.set_title("T07, 2017-08-10 – 2017-08-21 — record ends at the logged bearing failure", fontsize=8)
    ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False)
    ax.annotate("bearing failure\nlogged 08-20 06:08 —\nno further data\n(turbine offline)",
                xy=(bearing_ts, ax.get_ylim()[1] * 0.55), xytext=(8, 0), textcoords="offset points",
                fontsize=6, va="center")

    day_before = (ts >= bearing_ts - np.timedelta64(1, "D")) & (ts < bearing_ts)
    mean_24h = float(short_burn[day_before].mean()) if day_before.any() else float("nan")
    peak_idx = int(np.nanargmax(short_burn))
    peak_val = float(short_burn[peak_idx])
    peak_ts = ts[peak_idx]

    ax = axes[1]
    ax.plot(ts, short_burn, color="#B23A2E", lw=0.9, label="6h burn rate")
    ax.plot(ts, long_burn, color="#2C5F8A", lw=0.9, label="3d burn rate")
    ax.axhline(1.0, color="black", lw=0.6, ls=":")
    ax.text(bearing_ts - np.timedelta64(2, "D"), 1.0, "1.0× = sustainable rate ", fontsize=5.5, va="bottom", ha="right", color="#333333")
    ax.axvline(bearing_ts, color="black", lw=0.8, ls="--")
    ax.axvspan(record_end, bearing_ts + np.timedelta64(1, "D"), color="#dddddd", alpha=0.5)
    ax.axvspan(bearing_ts - np.timedelta64(1, "D"), bearing_ts, color="#B23A2E", alpha=0.08)
    ax.annotate(
        f"peak {peak_val:.1f}×", xy=(peak_ts, peak_val), xytext=(0, 6), textcoords="offset points",
        fontsize=6, ha="center", color="#B23A2E", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#B23A2E", lw=0.6),
    )
    ax.annotate(
        f"mean {mean_24h:.1f}×\nin 24h\nbefore\nfailure",
        xy=(bearing_ts - np.timedelta64(12, "h"), 25),
        fontsize=5.5, ha="center", va="center", color="#B23A2E",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
    )
    ax.set_ylabel("Burn rate (×)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=6, loc="upper left", frameon=False, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_xlim(ts.min(), bearing_ts + np.timedelta64(1, "D"))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_burn_rate_case_study.svg"); fig.savefig(FIG_DIR / "fig_burn_rate_case_study.png", dpi=220)
    plt.close(fig)
    print("wrote fig_burn_rate_case_study.svg")

    # print real numbers for the paper text
    day_before = (ts >= bearing_ts - np.timedelta64(1, "D")) & (ts < bearing_ts)
    print(f"data record ends at: {record_end} (bearing failure logged {bearing_ts}, {(bearing_ts - record_end)})")
    print(f"mean short burn rate, full window: {short_burn.mean():.2f}x")
    print(f"mean short burn rate, 24h before bearing failure: {short_burn[day_before].mean():.2f}x")
    print(f"peak short burn rate in window: {np.nanmax(short_burn):.2f}x")
    print(f"fraction non-compliant readings overall: {non_compliant.mean()*100:.1f}%")


if __name__ == "__main__":
    metrics = load_all_metrics()
    with open(METRICS_DIR / "all_metrics_combined.json", "w") as f:
        json.dump(metrics, f, indent=2)
    fig_accuracy_comparison(metrics)
    fig_cost_comparison(metrics)
    fig_power_curve()
    fig_burn_rate_case_study()
    print("\nAll figures written to", FIG_DIR)
