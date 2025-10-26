# masked_causes_conditional_shap_full.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from math import factorial

# -----------------------
# 1) Data: masked-causes
# -----------------------
rng = np.random.default_rng(42)
n = 1500

# Upstream cause
X1 = rng.normal(0, 1, n)

# Mediator: almost deterministic function of X1 (very small noise to strengthen masking)
X2 = 0.2 * X1 + rng.normal(0, 1, n)

# Mediator: almost deterministic function of X1 (very small noise to strengthen masking)
X3 = 0.5 * X2 + rng.normal(0, 1, n)

# Unrelated noise feature
X4 = rng.normal(0, 1, n)

# Target depends only on X2 (no direct X1 -> y path)
y = 3 * X3 + rng.normal(0, 0.1, n)

X = np.column_stack((X1, X2, X3, X4))
feat_names = ["X1", "X2", "X3", "X4"]
d = X.shape[1]

# ---------------------------------------
# 2) Predictive model (linear & stable)
# ---------------------------------------

"""
model = Ridge(alpha=1e-3, fit_intercept=True).fit(X, y)
b0 = float(model.intercept_)
b = model.coef_.astype(float)
"""

# # --- Oracle model (guarantees strongest masking) ---
b0 = 0.0
b = np.array([0.0, 0.0, 3.0, 0.0], dtype=float)

def f_pred_row(x_row):
    return b0 + float(np.dot(b, x_row))

# ------------------------------------------------------
# 3) Fit Gaussian to features for conditional expectations
# ------------------------------------------------------
mu = X.mean(axis=0)             # (d,)
Sigma = np.cov(X.T, bias=False) # (d,d)

def cond_mean(full_mu, full_Sigma, S_idx, x_S):
    """
    Conditional mean E[X_{-S} | X_S = x_S] for a multivariate normal.
    Returns indices of complement and the conditional mean vector (aligned to Sc order).
    """
    all_idx = np.arange(len(full_mu))
    S = np.array(sorted(S_idx), dtype=int)
    Sc = np.array([i for i in all_idx if i not in S], dtype=int)

    mu_S  = full_mu[S]
    mu_Sc = full_mu[Sc]

    Sig_SS  = full_Sigma[np.ix_(S,  S)]
    Sig_ScS = full_Sigma[np.ix_(Sc, S)]

    # E[X_Sc | X_S=x_S] = mu_Sc + Sig_ScS * Sig_SS^{-1} * (x_S - mu_S)
    mean_Sc = mu_Sc + Sig_ScS @ np.linalg.solve(Sig_SS, (x_S - mu_S))

    return Sc, mean_Sc

def conditional_expectation_f(x, S_idx):
    """E[f(X) | X_S = x_S] under Gaussian features + linear f."""
    S = np.array(sorted(S_idx), dtype=int)
    Sc, mean_Sc = cond_mean(mu, Sigma, S, x[S])
    x_exp = x.copy()
    x_exp[Sc] = mean_Sc
    return f_pred_row(x_exp)

def shapley_weight(d, s):
    return factorial(s) * factorial(d - s - 1) / factorial(d)

def conditional_shap_per_instance(x):
    """Exact conditional SHAP for a linear model (brute-force over all subsets)."""
    phi = np.zeros(d)
    F = np.arange(d)
    for i in range(d):
        others = [j for j in F if j != i]
        # enumerate all subsets of 'others'
        for mask in range(1 << (d - 1)):
            S = [others[k] for k in range(d - 1) if (mask >> k) & 1]
            w = shapley_weight(d, len(S))
            delta = conditional_expectation_f(x, S + [i]) - conditional_expectation_f(x, S)
            phi[i] += w * delta
    return phi

# Compute per-instance SHAP values (n x d)
phi = np.vstack([conditional_shap_per_instance(x) for x in X])

# Local accuracy check: phi sum + E[f] equals prediction
E_f = f_pred_row(mu)
pred = np.array([f_pred_row(row) for row in X])
assert np.allclose(phi.sum(axis=1) + E_f, pred, atol=1e-6)

# ----------------------------------------
# 4) Print mean absolute SHAP (global rank)
# ----------------------------------------
mean_abs_shap = np.abs(phi).mean(axis=0)
print("\nMean absolute SHAP values:")
for name, val in zip(feat_names, mean_abs_shap):
    print(f"{name}: {val:.4f}")

# ----------------------------------------
# 5) Beeswarm-style plot of ALL SHAP values
# ----------------------------------------
def beeswarm_shap(phi, X, feature_names, title):
    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    # vertical slots for features (top to bottom)
    y_positions = np.arange(len(feature_names))[::-1]  # put X1 on top
    jitter = 0.18
    rng_local = np.random.default_rng(0)

    # draw each feature's points
    for idx, (fname, y_base) in enumerate(zip(feature_names, y_positions)):
        vals = X[:, idx]
        # robust color scaling (1st-99th percentile)
        vmin, vmax = np.percentile(vals, [1, 99])
        vmin = min(vmin, vals.min())
        vmax = max(vmax, vals.max())
        norm = (vals - vmin) / (vmax - vmin + 1e-12)

        y_j = y_base + (rng_local.random(len(vals)) - 0.5) * 2 * jitter

        sc = ax.scatter(
            phi[:, idx], y_j, s=16,
            c=norm, cmap="coolwarm", alpha=0.9, linewidths=0
        )

    # cosmetics
    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_names, fontsize=11)
    ax.axvline(0.0, color="gray", linewidth=1)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Feature value", rotation=90)
    plt.tight_layout()
    plt.show()

beeswarm_shap(
    phi, X, feat_names,
    title="Conditional SHAP — masked causes demo (X1 -> X2 -> y; X3 unrelated)"
)
