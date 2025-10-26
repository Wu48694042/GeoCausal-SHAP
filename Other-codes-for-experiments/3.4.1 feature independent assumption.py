import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

# 1. Construct toy dataset
np.random.seed(42)
n = 500
X1 = np.random.normal(0, 1, size=n)
X2 = X1 + np.random.normal(0, 0.01, size=n)  # highly correlated
X3 = np.random.normal(0, 1, size=n)          # independent variable
y = 3 * X1 + np.random.normal(0, 0.1, size=n)  # only determined by X1

# 2. Create DataFrame
X = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})

# 3. Train model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X, y)

# 4. SHAP explanation
explainer = shap.Explainer(model)
shap_values = explainer(X)

# 5. Plot summary
shap.summary_plot(shap_values, X)

# 6. Print mean SHAP values
mean_shap = np.abs(shap_values.values).mean(axis=0)
shap_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean(|SHAP|)": mean_shap
}).sort_values(by="Mean(|SHAP|)", ascending=False)

print("\nMean(|SHAP|) ranking:")
print(shap_df)
