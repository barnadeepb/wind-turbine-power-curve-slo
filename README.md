# Turbine Power Curve as SLO

Wind turbine performance monitoring from real SCADA data, evaluated the way a
production engineering team evaluates a service: expected power output for
current conditions is treated as a service-level objective (SLO), and
sustained deviation is scored as a burn rate rather than a binary fault flag.

## Why

Wind turbine SCADA condition-monitoring datasets are typically extremely
imbalanced toward rare failure events (in this dataset: 28 confirmed
failures against ~3.3M ten-minute readings). Framing the problem as
continuous underperformance detection instead of rare-event classification
uses every row as a valid training example, and produces an interpretable,
operationally actionable signal (estimated energy loss) rather than a
generic anomaly score.

## Data

[EDP onshore wind farm SCADA dataset](https://doi.org/10.17632/zjxjnjp3xs.2)
(Kijanowski, Barszcz, Staszewski, Dao — AGH University of Krakow). 4
turbines (T01, T06, T07, T11), 2016-2017, 10-minute sampling. Not committed
to this repository — run `src/download_data.py` to fetch it locally.

## Method

1. IEC 61400-12 binned power curve (industry-standard baseline)
2. Parametric logistic curve fit
3. Random Forest
4. XGBoost
5. Heteroscedastic Gaussian Process (subsampled) — its predictive variance
   defines a condition-appropriate SLO tolerance band, rather than one
   hand-tuned global threshold
6. Small MLP
7. AutoML tabular regression (Google Vertex AI), as a build-vs-buy baseline

Train/test split is by calendar year (train 2016, test 2017) to avoid
leakage from autocorrelated adjacent readings and to keep full seasonal
coverage on both sides.

## Repository layout

```
src/            data download, preprocessing, modeling, evaluation
data/raw/       downloaded source files (gitignored)
data/processed/ cleaned/joined data (gitignored)
results/        metrics and figures used in the paper
paper/          IEEE conference paper
```

## Status

Work in progress.
