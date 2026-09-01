"""Small MLP power curve regressor, trained on GPU (Quadro T1000)."""

import numpy as np
import torch
from torch import nn


class PowerCurveMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 60, batch_size: int = 4096, patience: int = 8,
    log_every: int = 5,
):
    """Mini-batch training with early stopping on validation RMSE.
    Prints progress every `log_every` epochs so a stalled run is visible,
    not just silent.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PowerCurveMLP(X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = None
    stall = 0

    n = Xt.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_rmse = torch.sqrt(loss_fn(val_pred, yv)).item()

        if epoch % log_every == 0 or epoch == 1:
            print(f"  epoch {epoch}/{epochs}  val_rmse={val_rmse:.3f}")

        if val_rmse < best_val - 1e-4:
            best_val = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                print(f"  early stop at epoch {epoch} (no improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    return model, device
