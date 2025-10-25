from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def train_model(
    X,
    y,
    model_type: str = 'xgboost',
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    X_train: Optional[np.ndarray] = None,
    X_test:  Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    y_test:  Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[str, float]]:

    if X_train is None or y_train is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # === 2) Init model ===
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method="hist",
            objective="reg:squarederror",
        )
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif model_type == 'mlp':
        model = make_pipeline(
            StandardScaler(),
        MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=random_state,
        verbose=False,
        # silence training log
        )
    )
    else:
        raise ValueError("Unsupported model_type. Choose from ['xgboost', 'random_forest', 'mlp']")

    # === 3) Fit ===
    model.fit(X_train, y_train)

    # === 4) R² on train/test ===
    r2_train = float(model.score(X_train, y_train))
    r2_test = float(model.score(X_test, y_test)) if X_test is not None and y_test is not None else float('nan')

    # === 5) Global OOF R² via CV ===
    if model_type == 'xgboost':
        base = xgb.XGBRegressor(**model.get_params())
    elif model_type == 'random_forest':
        base = RandomForestRegressor(**model.get_params())
    elif model_type == 'mlp':
        mlp_estimator = model.named_steps['mlpregressor']
        base = make_pipeline(
            StandardScaler(),
            MLPRegressor(**mlp_estimator.get_params())
        )
    else:
        raise ValueError("Unsupported model_type for base model recreation.")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_oof = cross_val_predict(base, X, y, cv=cv, n_jobs=n_jobs, method="predict")
    r2_global_oof = float(r2_score(y, y_oof))

    metrics = {
        "r2_train": r2_train,
        "r2_test":  r2_test,
        "r2_global_oof": r2_global_oof,
        "n_train": int(len(y_train)),
        "n_test":  int(len(y_test)) if y_test is not None else 0,
        "cv_folds": int(n_splits),
        "model_type": model_type,
    }

    return model, metrics




#  powerset now returns lists instead of tuples
def do_intervention(x, S, model, baseline):
    if len(S) == 0:
        return model.predict(np.array(baseline).reshape(1, -1))[0]
    x_copy = np.array(baseline).copy()
    x_copy[list(S)] = x[list(S)]
    return model.predict(x_copy.reshape(1, -1))[0]


def evaluate_model(x, S, model, baseline):
    if len(S) == 0:
        return model.predict(np.array(baseline).reshape(1, -1))[0]
    x_copy = np.array(baseline).copy()
    x_copy[list(S)] = x[list(S)]
    return model.predict(x_copy.reshape(1, -1))[0]

