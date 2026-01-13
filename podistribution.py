import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# Device setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.manual_seed(42)

# ============================================================
# Parameters
# ============================================================
n_assets = 50
n_sets = 26
n_samples_mc = 150_000

w_max = 0.30          # max allocation per asset (box constraint)
lambda_reg = 0.05    # L2 regularization strength

# Storage: [return, risk, sharpe_reg, entropy]
observables = []

# ============================================================
# Generate random financial worlds
# ============================================================
for k in range(n_sets):

    # ----- Random expected returns -----
    mu = torch.empty(n_assets, device=device).uniform_(0.05, 0.20)

    # ----- Random positive-definite covariance -----
    A = torch.randn(n_assets, n_assets, device=device)
    Sigma = A @ A.T
    Sigma = Sigma / torch.max(torch.diag(Sigma))

    # ----- Monte Carlo portfolios with box constraint -----
    W = torch.rand(n_samples_mc, n_assets, device=device)
    W = W / W.sum(dim=1, keepdim=True)

    # Enforce max-weight constraint
    W = torch.clamp(W, max=w_max)
    W = W / W.sum(dim=1, keepdim=True)  # renormalize

    # Portfolio metrics
    returns = W @ mu
    variances = torch.einsum("bi,ij,bj->b", W, Sigma, W)
    risk = torch.sqrt(variances + 1e-12)

    # Regularized Sharpe objective
    l2_penalty = lambda_reg * torch.sum(W * W, dim=1)
    sharpe_reg = returns / risk - l2_penalty

    # Optimal portfolio
    best_idx = torch.argmax(sharpe_reg)
    w_star = W[best_idx]

    # ----- Portfolio observables -----
    port_return = (w_star @ mu).item()
    port_risk = torch.sqrt(w_star @ Sigma @ w_star).item()
    port_sharpe = port_return / port_risk

    eps = 1e-12
    entropy = -(w_star * torch.log(w_star + eps)).sum().item()

    observables.append([
        port_return,
        port_risk,
        port_sharpe,
        entropy
    ])

    print(
        f"Set {k+1:03d} | "
        f"Return={port_return:.3f}  "
        f"Risk={port_risk:.3f}  "
        f"Sharpe={port_sharpe:.3f}  "
        f"Entropy={entropy:.3f}"
    )

# ============================================================
# Convert to NumPy
# ============================================================
X = np.array(observables)
print("\nObservable matrix shape:", X.shape)

# ============================================================
# 3D Observable-Space Geometry (MAIN RESULT)
# ============================================================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    X[:, 1],        # Risk
    X[:, 0],        # Return
    X[:, 3],        # Entropy
    c=X[:, 2],      # Sharpe
    cmap="plasma",
    s=70,
    edgecolors="k"
)

ax.set_xlabel("Portfolio Risk")
ax.set_ylabel("Portfolio Return")
ax.set_zlabel("Portfolio Entropy")

fig.colorbar(sc, ax=ax, label="Sharpe Ratio")
ax.set_title("3D Geometry with Minimal Constraints")

plt.tight_layout()
plt.show()
