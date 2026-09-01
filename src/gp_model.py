"""Heteroscedastic Gaussian Process for the power curve, via the Most Likely
Heteroscedastic GP approach (Kersting et al., ICML 2007): fit a homoskedastic
GP, estimate input-dependent noise from its residuals with a second GP, then
refit the main GP with that noise fixed per-point. The result's predictive
variance is what defines the condition-appropriate SLO tolerance band from
the plan, rather than one hand-tuned global threshold.

Runs on a stratified subsample (not the full training set) — exact GP
inference is O(n^3); at full dataset size that's not slow, it's impossible.
"""

import numpy as np
import torch
import gpytorch


class _ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def stratified_subsample(wind_speed: np.ndarray, n_target: int, bin_width: float = 0.5, seed: int = 0):
    """Sample roughly evenly across wind-speed bins so the GP sees the full
    operating envelope, not just the densest region."""
    rng = np.random.default_rng(seed)
    bins = np.digitize(wind_speed, np.arange(0, wind_speed.max() + bin_width, bin_width))
    idx_by_bin = {}
    for i, b in enumerate(bins):
        idx_by_bin.setdefault(b, []).append(i)
    n_bins = len(idx_by_bin)
    per_bin = max(1, n_target // n_bins)
    chosen = []
    for b, idxs in idx_by_bin.items():
        take = min(per_bin, len(idxs))
        chosen.extend(rng.choice(idxs, size=take, replace=False))
    chosen = np.array(chosen)
    if len(chosen) > n_target:
        chosen = rng.choice(chosen, size=n_target, replace=False)
    return chosen


def _fit_exact_gp(x, y, likelihood, iters=150, lr=0.05):
    model = _ExactGP(x, y, likelihood)
    model.train()
    likelihood.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(iters):
        opt.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        loss.backward()
        opt.step()
        if (i + 1) % 25 == 0:
            print(f"    gp iter {i + 1}/{iters}  loss={loss.item():.3f}")
    model.eval()
    likelihood.eval()
    return model


def fit_heteroscedastic_gp(wind_speed: np.ndarray, power: np.ndarray, n_subsample: int = 15000):
    device = "cpu"  # exact GP at this scale — CPU is simpler and plenty fast
    idx = stratified_subsample(wind_speed, n_subsample)
    x = torch.tensor(wind_speed[idx], dtype=torch.float64, device=device).unsqueeze(-1)
    y = torch.tensor(power[idx], dtype=torch.float64, device=device)
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std

    print("  stage 1: homoskedastic GP")
    lik1 = gpytorch.likelihoods.GaussianLikelihood().double()
    model1 = _fit_exact_gp(x, y_norm, lik1)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred1 = lik1(model1(x)).mean
    resid_sq = (y_norm - pred1) ** 2
    # floor to avoid log(0); noise model works in log-variance space
    log_noise_target = torch.log(resid_sq.clamp_min(1e-4))

    print("  stage 2: noise-level GP (models log residual variance vs wind speed)")
    lik2 = gpytorch.likelihoods.GaussianLikelihood().double()
    noise_model = _fit_exact_gp(x, log_noise_target, lik2, iters=100)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        est_log_noise = lik2(noise_model(x)).mean
    est_noise = torch.exp(est_log_noise).clamp_min(1e-4)

    print("  stage 3: refit main GP with fixed heteroscedastic noise")
    lik3 = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=est_noise).double()
    model3 = _fit_exact_gp(x, y_norm, lik3)

    return {
        "mean_model": model3, "mean_likelihood": lik3,
        "noise_model": noise_model, "noise_likelihood": lik2,
        "y_mean": y_mean.item(), "y_std": y_std.item(),
        "n_train": len(idx),
    }


def predict(fitted, wind_speed: np.ndarray):
    """Returns (mean_power, std_power) — std is the condition-appropriate SLO band."""
    x = torch.tensor(wind_speed, dtype=torch.float64).unsqueeze(-1)
    model, lik = fitted["mean_model"], fitted["mean_likelihood"]
    noise_model, noise_lik = fitted["noise_model"], fitted["noise_likelihood"]
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        mean_norm = lik(model(x)).mean
        log_noise = noise_lik(noise_model(x)).mean
    mean = mean_norm * fitted["y_std"] + fitted["y_mean"]
    std = torch.sqrt(torch.exp(log_noise)) * fitted["y_std"]
    return mean.numpy(), std.numpy()
