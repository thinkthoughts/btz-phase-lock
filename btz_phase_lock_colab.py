# BTZ Phase-Lock Demo (Colab-ready)
# ------------------------------------------------------------
# Minimal, runnable toy demo inspired by arXiv:2604.05970.
# This is a simplified BTZ reconstruction notebook script:
# - two tanh MLPs
# - alternating Adam updates
# - EE + WL + regularization + 45° balance penalty
# - synthetic BTZ-like targets for quick experimentation
#
# Notes:
# 1) This is a compact demo, not a full reproduction of the paper.
# 2) It is meant to generate shareable plots / a quick Colab link.
# 3) You can paste this whole file into a Colab notebook cell and run.

# %% [markdown]
# # 1. Setup

# %%
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# %% [markdown]
# # 2. Toy BTZ background and synthetic data
#
# We use a simple non-rotating BTZ-like blackening factor
#     f(r) = r^2 - r_h^2
# with AdS radius set to 1.
#
# For a quick demo, instead of solving the full geodesic / Nambu-Goto system,
# we use smooth synthetic observables that depend on f(r) through a turning-point
# proxy r_*(ell). This keeps the code light and trainable in Colab.

# %%
R_ADS = 1.0
R_H = 1.0
ELL_MIN = 0.1
ELL_MAX = math.pi * 1.0
R_MIN = 1.05 * R_H
R_MAX = 3.0


def f_btz(r: torch.Tensor) -> torch.Tensor:
    return r**2 - R_H**2


def r_star_from_ell(ell: torch.Tensor) -> torch.Tensor:
    # Smooth monotone proxy: small ell -> large r_*, large ell -> near horizon
    # Keeps the toy problem well-behaved.
    return R_H + 0.35 + 2.2 / (ell + 0.35)


def s_ee_true(ell: torch.Tensor) -> torch.Tensor:
    # Synthetic EE target, roughly log-like with f(r_*) dependence.
    rs = r_star_from_ell(ell)
    return torch.log(1.0 + 3.0 * ell) + 0.08 / torch.sqrt(f_btz(rs) + 1e-6)


def w_wl_true(ell: torch.Tensor) -> torch.Tensor:
    # Synthetic WL target, chosen to depend differently on f(r_*).
    rs = r_star_from_ell(ell)
    return torch.sqrt(ell + 0.2) + 0.16 / (f_btz(rs) + 0.2)


N_ELL = 160
ell_grid = torch.linspace(ELL_MIN, ELL_MAX, N_ELL, device=device).view(-1, 1)
s_ee_obs = s_ee_true(ell_grid)
w_wl_obs = w_wl_true(ell_grid)

r_plot = torch.linspace(R_MIN, R_MAX, 256, device=device).view(-1, 1)
f_true_plot = f_btz(r_plot)

# %% [markdown]
# # 3. Models
#
# - LModel predicts an effective turning-point / latent geometry variable from ell.
# - VModel predicts the blackening factor f(r).
#
# Both use tanh MLPs, matching the lightweight architecture style used in the paper.

# %%
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 32, depth: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            layers.append(nn.Tanh())
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LModel(nn.Module):
    # ell -> latent turning point r_hat(ell)
    def __init__(self, width: int = 32, depth: int = 2):
        super().__init__()
        self.core = MLP(1, 1, width=width, depth=depth)

    def forward(self, ell: torch.Tensor) -> torch.Tensor:
        # enforce r_hat > horizon via softplus offset
        raw = self.core(ell)
        return R_H + 0.05 + F.softplus(raw)


class VModel(nn.Module):
    # r -> blackening factor f_hat(r)
    def __init__(self, width: int = 20, depth: int = 2):
        super().__init__()
        self.core = MLP(1, 1, width=width, depth=depth)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # positive blackening factor outside horizon
        raw = self.core(r)
        return F.softplus(raw)


L_net = LModel(width=32, depth=2).to(device)
V_net = VModel(width=20, depth=2).to(device)

# %% [markdown]
# # 4. Observable heads and losses
#
# We define differentiable toy forward maps:
# - EE head depends on ell and f_hat(r_hat(ell))
# - WL head depends on ell and f_hat(r_hat(ell)) in a different way
#
# The 45° penalty nudges balanced EE/WL contributions, acting as a phase-lock regularizer.

# %%

def s_ee_pred(ell: torch.Tensor, r_hat: torch.Tensor, f_hat_at_rhat: torch.Tensor) -> torch.Tensor:
    return torch.log(1.0 + 3.0 * ell) + 0.08 / torch.sqrt(f_hat_at_rhat + 1e-6)


def w_wl_pred(ell: torch.Tensor, r_hat: torch.Tensor, f_hat_at_rhat: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(ell + 0.2) + 0.16 / (f_hat_at_rhat + 0.2)


def smoothness_penalty(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    return (grad**2).mean()


def monotonicity_penalty(r_hat: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(r_hat.sum(), ell, create_graph=True)[0]
    # r_hat should generally decrease as ell grows
    return F.relu(grad).mean()


def phase_lock_penalty(loss_ee: torch.Tensor, loss_wl: torch.Tensor) -> torch.Tensor:
    # Encourage balanced contributions near 45°: cos(theta)=1/sqrt(2)
    # For positive scalars a,b, equal weighting gives 45° in the (a,b) plane.
    a = torch.sqrt(loss_ee + 1e-12)
    b = torch.sqrt(loss_wl + 1e-12)
    ratio = a / (b + 1e-12)
    return (torch.log(ratio) ** 2)

# %% [markdown]
# # 5. Training setup

# %%
@dataclass
class TrainConfig:
    epochs: int = 500
    lr_L: float = 2e-3
    lr_V: float = 2e-3
    w_ee: float = 1.0
    w_wl: float = 1.0
    w_smooth: float = 2e-4
    w_mono: float = 5e-3
    w_phase: float = 2e-2
    print_every: int = 50


cfg = TrainConfig()
opt_L = torch.optim.Adam(L_net.parameters(), lr=cfg.lr_L)
opt_V = torch.optim.Adam(V_net.parameters(), lr=cfg.lr_V)

history = {
    "loss_total": [],
    "loss_ee": [],
    "loss_wl": [],
    "loss_phase": [],
    "metric_rel_err": [],
}

# %% [markdown]
# # 6. Alternating training loop

# %%
for epoch in range(1, cfg.epochs + 1):
    ell = ell_grid.clone().detach().requires_grad_(True)

    # ------------------
    # L-step: update latent geometry model L
    # ------------------
    opt_L.zero_grad()
    r_hat = L_net(ell)
    f_hat_at_rhat = V_net(r_hat)

    ee_hat = s_ee_pred(ell, r_hat, f_hat_at_rhat)
    wl_hat = w_wl_pred(ell, r_hat, f_hat_at_rhat)

    loss_ee_L = F.mse_loss(ee_hat, s_ee_obs)
    loss_wl_L = F.mse_loss(wl_hat, w_wl_obs)
    loss_mono_L = monotonicity_penalty(r_hat, ell)
    loss_phase_L = phase_lock_penalty(loss_ee_L, loss_wl_L)

    loss_L = (
        cfg.w_ee * loss_ee_L
        + cfg.w_wl * loss_wl_L
        + cfg.w_mono * loss_mono_L
        + cfg.w_phase * loss_phase_L
    )
    loss_L.backward()
    opt_L.step()

    # ------------------
    # V-step: update metric model V
    # ------------------
    opt_V.zero_grad()
    ell = ell_grid.clone().detach().requires_grad_(True)
    r_hat = L_net(ell).detach()  # alternate: keep L fixed here
    r_hat.requires_grad_(True)

    f_hat_at_rhat = V_net(r_hat)
    ee_hat = s_ee_pred(ell, r_hat, f_hat_at_rhat)
    wl_hat = w_wl_pred(ell, r_hat, f_hat_at_rhat)

    loss_ee_V = F.mse_loss(ee_hat, s_ee_obs)
    loss_wl_V = F.mse_loss(wl_hat, w_wl_obs)
    loss_smooth_V = smoothness_penalty(f_hat_at_rhat, r_hat)
    loss_phase_V = phase_lock_penalty(loss_ee_V, loss_wl_V)

    loss_V = (
        cfg.w_ee * loss_ee_V
        + cfg.w_wl * loss_wl_V
        + cfg.w_smooth * loss_smooth_V
        + cfg.w_phase * loss_phase_V
    )
    loss_V.backward()
    opt_V.step()

    # ------------------
    # Logging
    # ------------------
    with torch.no_grad():
        f_pred_plot = V_net(r_plot)
        rel_err = ((f_pred_plot - f_true_plot).abs() / (f_true_plot.abs() + 1e-6)).mean().item() * 100.0
        total_loss = (loss_L + loss_V).item()

        history["loss_total"].append(total_loss)
        history["loss_ee"].append((loss_ee_V.item() + loss_ee_L.item()) / 2.0)
        history["loss_wl"].append((loss_wl_V.item() + loss_wl_L.item()) / 2.0)
        history["loss_phase"].append((loss_phase_V.item() + loss_phase_L.item()) / 2.0)
        history["metric_rel_err"].append(rel_err)

    if epoch % cfg.print_every == 0 or epoch == 1:
        print(
            f"epoch {epoch:4d} | total={total_loss:.6f} | "
            f"ee={history['loss_ee'][-1]:.6f} | wl={history['loss_wl'][-1]:.6f} | "
            f"phase={history['loss_phase'][-1]:.6f} | rel_err={rel_err:.3f}%"
        )

# %% [markdown]
# # 7. Final plots

# %%
with torch.no_grad():
    r_hat_final = L_net(ell_grid)
    f_hat_rhat_final = V_net(r_hat_final)
    ee_pred_final = s_ee_pred(ell_grid, r_hat_final, f_hat_rhat_final)
    wl_pred_final = w_wl_pred(ell_grid, r_hat_final, f_hat_rhat_final)
    f_pred_plot = V_net(r_plot)

plt.figure(figsize=(6, 4))
plt.plot(ell_grid.cpu(), s_ee_obs.cpu(), label="EE true")
plt.plot(ell_grid.cpu(), ee_pred_final.cpu(), "--", label="EE pred")
plt.xlabel("interval size ℓ")
plt.ylabel("S_EE(ℓ)")
plt.title("BTZ phase-lock demo: EE curve")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(ell_grid.cpu(), w_wl_obs.cpu(), label="WL true")
plt.plot(ell_grid.cpu(), wl_pred_final.cpu(), "--", label="WL pred")
plt.xlabel("probe size ℓ")
plt.ylabel("W_WL(ℓ)")
plt.title("BTZ phase-lock demo: WL curve")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(r_plot.cpu(), f_true_plot.cpu(), label="BTZ true")
plt.plot(r_plot.cpu(), f_pred_plot.cpu(), "--", label="Learned")
plt.xlabel("r")
plt.ylabel("f(r)")
plt.title("Recovered blackening factor")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(history["loss_total"], label="total loss")
plt.plot(history["loss_ee"], label="EE loss")
plt.plot(history["loss_wl"], label="WL loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Training losses")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(history["metric_rel_err"])
plt.xlabel("epoch")
plt.ylabel("mean relative error (%)")
plt.title("Metric reconstruction error")
plt.tight_layout()
plt.show()

# %% [markdown]
# # 8. Save a shareable figure bundle (optional)

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

axes[0].plot(ell_grid.cpu(), s_ee_obs.cpu(), label="EE true")
axes[0].plot(ell_grid.cpu(), ee_pred_final.cpu(), "--", label="EE pred")
axes[0].set_title("EE")
axes[0].set_xlabel("ℓ")
axes[0].legend()

axes[1].plot(ell_grid.cpu(), w_wl_obs.cpu(), label="WL true")
axes[1].plot(ell_grid.cpu(), wl_pred_final.cpu(), "--", label="WL pred")
axes[1].set_title("WL")
axes[1].set_xlabel("ℓ")
axes[1].legend()

axes[2].plot(r_plot.cpu(), f_true_plot.cpu(), label="BTZ true")
axes[2].plot(r_plot.cpu(), f_pred_plot.cpu(), "--", label="Learned")
axes[2].set_title("Blackening factor")
axes[2].set_xlabel("r")
axes[2].legend()

axes[3].plot(history["metric_rel_err"], label="rel err %")
axes[3].set_title("Metric error")
axes[3].set_xlabel("epoch")
axes[3].legend()

plt.suptitle("BTZ Phase-Lock Demo")
plt.tight_layout()
plt.savefig("btz_phase_lock_demo.png", dpi=180, bbox_inches="tight")
plt.show()

print("Saved: btz_phase_lock_demo.png")

# %% [markdown]
# # 9. Suggested tweet / notes
#
# - This is a toy BTZ Colab skeleton inspired by the paper.
# - It demonstrates the dual-network + alternating updates pattern.
# - For a closer reproduction, replace the synthetic observables with the full
#   RT / Nambu-Goto variational functionals and match the paper's geometry setup.
