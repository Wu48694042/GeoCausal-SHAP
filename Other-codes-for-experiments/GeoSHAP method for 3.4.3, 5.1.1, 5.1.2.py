
import os
import time
from time import perf_counter
from datetime import datetime
try:
    # Python 3.9+: use system tz database
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Europe/Berlin")
except Exception:
    TZ = None  # fallback to naive datetime if zoneinfo is unavailable

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from geoshapley import GeoShapleyExplainer
from utils import train_model
from sklearn.model_selection import train_test_split

# ----------------------- Config -----------------------
# CSV_PATH = "D:/TUM Master GuG/SS2025/Master's thesis - Geodesy and Geoinformation/dataset/melb_data.csv"  # change if running locally
# SPATIAL_COLS = ["Lattitude", "Longtitude"]  # for melb_data.csv
# TARGET_COL = "Price"
CSV_PATH = "D:/TUM Master GuG/SS2025/Master's thesis - Geodesy and Geoinformation/dataset/seattle_sample_all.csv"
SPATIAL_COLS = ["UTM_X", "UTM_Y"]  # for seattle_sample_all.csv
TARGET_COL = "log_price"  # set to your target column name if known, e.g., "price"; None => auto-pick numeric last

RANDOM_STATE = 42
N_TREES = 400

# ------------------ Small utilities -------------------
def now_str():
    dt = datetime.now(TZ) if TZ else datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z") if TZ else dt.strftime("%Y-%m-%d %H:%M:%S")

def pick_target_column(df, spatial_cols, preferred=None):
    """Pick a numeric target column. Priority: preferred -> 'target' -> last numeric not in spatial."""
    if preferred and preferred in df.columns:
        return preferred
    if "target" in df.columns:
        return "target"
    # choose the last numeric column not spatial
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = [c for c in num_cols if c not in spatial_cols]
    if not candidates:
        raise ValueError("No numeric target candidates found. Please set TARGET_COL to an existing numeric column.")
    return candidates[-1]

class ArrayWrapper:
    """Wrap a numpy array so geoshapley can call both .values and [:, :]."""
    def __init__(self, arr):
        self._arr = np.asarray(arr)
    @property
    def values(self):
        return self._arr
    @property
    def shape(self):
        return self._arr.shape
    @property
    def ndim(self):
        return self._arr.ndim
    def __len__(self):
        return len(self._arr)
    def __getitem__(self, idx):
        return self._arr[idx]
    def __array__(self, *args, **kwargs):
        return self._arr

def extract_feature_shap_matrix(res, X_df):
    """Find a 2D (n_samples x n_features) feature SHAP-like matrix stored in result."""
    for name in ["values", "shap_values", "phi", "Phi", "total", "geoshapley_values"]:
        if hasattr(res, name):
            arr = getattr(res, name)
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] == len(X_df):
                return arr
    for name, val in vars(res).items():
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[0] == len(X_df):
            return val
    raise AttributeError("GeoShapleyResults: cannot find per-instance feature SHAP matrix.")

# ---------------------- Main flow ---------------------
start_time = time.time()
script_start = perf_counter()
print(f"[{now_str()}] Script started.")
print(f"CSV path: {CSV_PATH}")

load_t0 = perf_counter()
df = pd.read_csv(CSV_PATH)
"""
df = df[[
    "Lattitude",
    "Longtitude",
    "Rooms",
    "Distance",
    "Bathroom",
    "Car",
    "BuildingArea",
    "YearBuilt",
    "Price"
]]

df["YearBuilt"] = df["YearBuilt"].astype(float)

df["YearBuilt"] = 2025 - df["YearBuilt"]
"""
print(f"Loaded CSV with shape={df.shape} and columns={list(df.columns)}")
load_t1 = perf_counter()

# Basic checks
for c in SPATIAL_COLS:
    if c not in df.columns:
        raise KeyError(f"Required spatial column '{c}' not found in CSV.")

# Auto-pick or use the provided target col
y_col = pick_target_column(df, spatial_cols=SPATIAL_COLS, preferred=TARGET_COL)
print(f"Using target column: {y_col}")

# Build feature matrix: all numeric, excluding target; ensure spatial are at the end
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
feat_candidates = [c for c in numeric_cols if c not in SPATIAL_COLS and c != y_col]
print(feat_candidates)
if not feat_candidates:
    raise ValueError("No numeric feature columns (excluding target and spatial) found.")

# Order: [non-spatial features..., UTM_X, UTM_Y]
ordered_cols = feat_candidates + SPATIAL_COLS
X_geo_full = df[ordered_cols].copy()
y_full = df[y_col].copy()

# Drop rows with missing values on used columns
mask = X_geo_full.notna().all(axis=1) & y_full.notna()
X_geo_full = X_geo_full[mask]
y_full = y_full[mask]
print(f"After dropping NA rows, X shape={X_geo_full.shape}, y shape={y_full.shape}")

# Train model on the full cleaned dataset
train_t0 = perf_counter()
X_tr, X_te, y_tr, y_te = train_test_split(X_geo_full, y_full, test_size=0.2, random_state=42)
model, metrics = train_model(
    X_geo_full, y_full,
    model_type="xgboost",
    test_size=0.2,
    n_splits=5,
    random_state=42,
    n_jobs=-1,
    X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te
)
print(f"R^2 (train split): {metrics['r2_train']:.4f}")
print(f"R^2 (test  split): {metrics['r2_test']:.4f}")
print(f"R^2 (global OOF):  {metrics['r2_global_oof']:.4f}")

# Prepare background samples (from the full set) and ALL rows to explain
BG_SIZE = min(30, len(X_geo_full))
background_np = X_geo_full.sample(BG_SIZE, random_state=RANDOM_STATE).values

# Now explain the full dataset
X_to_explain = X_geo_full.copy()
print(f"Explaining ALL {len(X_to_explain)} rows. X_to_explain shape={X_to_explain.shape}")

# Wrap arrays for GeoShapley
bg_wrapped = ArrayWrapper(background_np)
X_wrapped = ArrayWrapper(X_to_explain.values)

# Compute GeoShapley
explain_t0 = perf_counter()
explainer = GeoShapleyExplainer(model.predict, bg_wrapped)
result = explainer.explain(X_wrapped, n_jobs=1)  # use single process first for stability
explain_t1 = perf_counter()
print(f"[{now_str()}] Explain done. Elapsed {(explain_t1 - explain_t0):.2f}s")

# For geoshapley's internal plotting helpers
result.X_geo = X_to_explain

# ---- Optional built-in visuals (may be unavailable depending on package version) ----
try:
    result.summary_plot()
except Exception as e:
    print(f"summary_plot() failed: {e}")
try:
    result.partial_dependence_plots()
except Exception as e:
    print(f"partial_dependence_plots() failed: {e}")

# --------- Compose contributions: features + GEO (no x GEO interaction bars) ---------
feat_names = list(X_to_explain.columns[:-2])  # non-spatial feature names
feat_shap = extract_feature_shap_matrix(result, X_to_explain)  # (n_samples, n_features)

# Spatial interaction contributions (feature x GEO); if missing, treat as zeros
try:
    svc_df = result.get_svc()                       # DataFrame, columns aligned with feat_names
    inter_vals = svc_df.values
except Exception:
    inter_vals = np.zeros_like(feat_shap)

# Expected model output under background
try:
    bg = getattr(result, "background", None)
    if bg is None:
        raise AttributeError
    bg_vals = np.asarray(getattr(bg, "values", bg))
    expected_f = model.predict(bg_vals).mean()
except Exception:
    expected_f = model.predict(background_np).mean()

# GEO direct contribution via additivity
pred = model.predict(X_to_explain)
geo_phi = pred - expected_f - feat_shap.sum(axis=1) - inter_vals.sum(axis=1)

# Global importance for "features + GEO" only (no x GEO bars)
feat_mean_abs = np.abs(feat_shap).mean(axis=0)   # per-feature
geo_mean_abs  = float(np.mean(np.abs(geo_phi)))  # GEO
names_plot = feat_names + ["GEO"]
vals_plot  = list(feat_mean_abs) + [geo_mean_abs]

# Plot horizontal bar chart
order = np.argsort(vals_plot)[::-1]
plt.figure(figsize=(8, 5))
plt.barh([names_plot[i] for i in order], [vals_plot[i] for i in order])
plt.gca().invert_yaxis()
plt.xlabel("Mean |GeoShapley Value|")
plt.title("Global Importance (features + GEO, first 5 samples)")
plt.tight_layout()
plt.show()

# Print sorted contributions (features + GEO only)
print("\nMean absolute GeoShapley contributions (features + GEO; first 5 samples):")
for name, val in sorted(zip(names_plot, vals_plot), key=lambda x: -x[1]):
    print(f"{name:>20s}: {val:.4f}")

script_end = perf_counter()
print(f"\n[{now_str()}] Script finished. Total elapsed {(script_end - script_start):.2f}s")

end_time = time.time()
elapsed = end_time - start_time
print(f"\nTotal runtime: {elapsed:.2f} seconds")