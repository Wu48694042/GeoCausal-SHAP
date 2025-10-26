import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import networkx as nx
from Causal_discovery import learn_causal_graph, visualize_dag
from Shap_explainer import GeoCausalSHAP
from utils import train_model
import time
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# === 1. Load dataset from CSV (seattle housing price dataset)===
"""
data = pd.read_csv(
    "D:/TUM Master GuG/SS2025/Master's thesis - Geodesy and Geoinformation/dataset/seattle_sample_all.csv")

# Separate features and target
y = data["log_price"].values
X = data.drop(columns=["log_price"])
feature_names = X.columns.tolist()
"""

# === 1. Load Europe Climate dataset (Munich temperature)===
"""
df = pd.read_excel("D:/TUM Master GuG/SS2025/Master's thesis - Geodesy and Geoinformation/dataset/Munich_Mean Temperature.xlsx")

# Unit conversion
df["mean temperature in 0.1"] /= 10
df["precipitation amount in 0.1 mm"] /= 10
df["sunshine in 0.1 Hours"] /= 10
df["wind gust in 0.1 m/s"] /= 10
df["wind speed in 0.1 m/s"] /= 10

# Construct feature X and target variable Y
X = df[[
    "global radiation in W/m2",
    "precipitation amount in 0.1 mm",
    "sunshine in 0.1 Hours",
    "snow depth in 1 cm",
    "humidity in 1 %",
    "wind gust in 0.1 m/s",
    "wind speed in 0.1 m/s"
]]

y = df["mean temperature in 0.1"].values

# Clean the name of variables
X.columns = [
    "radiation", "precipitation", "sunshine", "snow_depth",
    "humidity", "wind_gust", "wind_speed"
]

feature_names = X.columns.tolist()
"""

#=== 1. Load Tiantan Air Quality dataset===
"""
df = pd.read_csv("D:/TUM Master GuG/SS2025/Master's thesis - Geodesy and Geoinformation/dataset/TiantanAQ.csv")

# Construct feature X and target variable Y
X = df[[
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]]

y = df["PM2.5"].values

feature_names = X.columns.tolist()
"""

# === 1. Load Melbourne Housing dataset===
"""
df = pd.read_csv(
    "D:/TUM Master GuG/SS2025/Master's thesis - Geodesy and Geoinformation/dataset/melb_data.csv")

df = df.dropna(subset=["Car", "BuildingArea", "YearBuilt"])

df["YearBuilt"] = 2025 - df["YearBuilt"]

# Construct feature X and target variable Y
X = df[[
    "Lattitude",  # Spelling mistake from dataset, not from author of the thesis
    "Longtitude",
    "Rooms",
    "Distance",
    "Bathroom",
    "Car",
    "BuildingArea",
    "YearBuilt"
]]

y = df["Price"].values

feature_names = X.columns.tolist()
"""

# === 1. Load Qinghai-Tibet Plateau Urban Region Frozen Ground Dataset===
"""
data = pd.read_csv("D:/TUM Master GuG/SS2025/Master's thesis - Geodesy and Geoinformation/dataset/Newdata_2020.csv")

X = data[['CLCD', 'long', 'lat', 'Wind', 'Tem', 'Sun', 'Sand', 'Clay', 'Silt', 'Slope',
          'Relief', 'Pre', 'NDVI', 'Night', 'GPP',
          'ET', 'Erosion', 'Disturb', 'Water', 'Rail', 'Road',
          'Rock', 'DEM', 'Aspect']]

y = data['Frozen2005']

feature_names = X.columns.tolist()
"""

# === 1. Load Munich Population Distribution Dataset===
"""
data = pd.read_csv("D:/TUM Master GuG/SS2025/Master's thesis - Geodesy and Geoinformation/dataset/data_100m_withlocation.csv")

X = data[[
    # Remote Sensing
    'ntl_mean', 'ndvi_mean',

    # Building and landscape
    'build_area', 'volume',

    # Land use
    'p_Industri', 'p_urban_fa', 'p_Green_ur', 'p_forests',
    'p_farmland', 'p_Land_wit', 'p_Sports_a', 'p_road___t',
    'p_Water', 'p_Construc', 'p_Mineral', 'p_Isolated',

    # Point of interests (800m)
    'acco800m', 'airp800m', 'cult800m', 'educ800m',
    'rest800m', 'gove800m', 'heli800m', 'leis800m',
    'life800m', 'heal800m', 'park800m', 'pubt800m',
    'rail800m', 'recy800m', 'reso800m', 'reta800m',
    'spor800m',

    # locational variables
    'Long', 'Lat'
]]

y = data['PopDensity']
"""

SEED = 42
np.random.seed(SEED)

# ==== Config ====
MODEL_TYPE = "random_forest"  # 'xgboost' | 'random_forest' | 'mlp'
DAG_METHOD = "pc"  # 'notears' | 'pc' | 'ges'
SHAP_MODE = "mode 1"  # 'mode 1' | 'mode 2'
BACKGROUND_N = 30  # background set size
TEST_SIZE = 0.2  # for reporting train/test R^2

start = time.perf_counter()

# === 1. Load Dataset===

# change the address as your setting, dear users.
data = pd.read_csv(
    "D:/TUM Master GuG/SS2025/Master's thesis - Geodesy and Geoinformation/dataset/seattle_sample_all.csv")

# Separate features and target
y = data["log_price"].values
X = data.drop(columns=["log_price"])
feature_names = X.columns.tolist()
print(feature_names)
n, p = X.shape
print(n)

# 2) External split for reporting R^2
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
print("split finish!")
print("wait for prediction")

# 3) Train using your updated train_model WITH the external split
model, metrics = train_model(
    X, y,
    model_type=MODEL_TYPE,
    test_size=TEST_SIZE,
    n_splits=5,
    random_state=SEED,
    n_jobs=-1,
    X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te
)

print(f"R^2 (train split): {metrics['r2_train']:.4f}")
print(f"R^2 (test  split): {metrics['r2_test']:.4f}")
print(f"R^2 (global OOF):  {metrics['r2_global_oof']:.4f}")

# ==== 3. Learn one global DAG ====
G = learn_causal_graph(X, method=DAG_METHOD)
print(nx.is_directed_acyclic_graph(G))
visualize_dag(G, feature_names=feature_names)

# ==== 4. Initialize explainer and background dataset ====
geoshap = GeoCausalSHAP(G)
background_data = X.sample(n=min(BACKGROUND_N, len(X)), random_state=SEED).values

# ==== 5. Select all samples to explain ====
X_array = X.values
explain_idx = np.arange(n)
phi_matrix = np.zeros((n, p))

# ==== 6. Compute SHAP for the selected samples ====
for i, idx in enumerate(tqdm(explain_idx, desc="GeoCausal SHAP (all samples)", ncols=100)):
    x_i = X_array[idx]
    phi_matrix[i] = geoshap.explain_instance(x_i, model, background_data, mode=SHAP_MODE)

# ==== 7. Global mean absolute SHAP ====
mean_abs_phi = np.abs(phi_matrix).mean(axis=0)
print("\n=== Global mean(|GeoCausal SHAP|) over all samples ===")
for name, value in zip(feature_names, mean_abs_phi):
    print(f"  {name:20s}: {value:.4f}")

elapsed = time.perf_counter() - start
print(f"\nTotal runtime: {elapsed:.2f} s")

# === 8. Visualize SHAP spatial distribution for one feature ===

for fname in feature_names:
    if fname in ["Long", "Lat"]:
        continue
    idx = feature_names.index(fname)
    shap_vals = phi_matrix[:, idx]

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(X["Long"], X["Lat"], c=shap_vals, cmap='coolwarm', s=20, edgecolor='k', alpha=0.5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Spatial Distribution of SHAP Values for '{fname}'")
    cbar = plt.colorbar(sc)
    cbar.set_label("SHAP Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

